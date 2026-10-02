"""
Analisis contoh generasi dan metrik leksikal checkpoint GPT.
"""
import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import get_config
from checkpoint_meta import validate_checkpoint_metadata
from embedding import build_tokenizer_and_embedding
from gptModel import GPTModel
from generate_gpt import (
    _resolve_device,
    _set_seed,
    _load_checkpoint_payload,
    _resolve_model_state,
    _checkpoint_env,
    _sample_next_token,
)

# ============================================================
# Sampling Configuration
# ============================================================
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.95
REPETITION_PENALTY = 1.1
MAX_NEW_TOKENS = 80

# ============================================================
# Helper: generic sequence functions
# ============================================================
def tokenize_for_metrics(text: str):
    """Simplistic tokenization for metric computation."""
    text = text.lower()
    words = re.findall(r"[a-z0-9\u00c0-\u024f]+", text)
    return words


# ============================================================
# Core generate (reuse generate_gpt.generate_text logic)
# ============================================================
def generate_list(model, tokenizer, prompt, n=5, device="cpu",
                  temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P,
                  penalty=REPETITION_PENALTY, max_new_tokens=MAX_NEW_TOKENS,
                  seeds=None):
    outputs = []
    for i in range(n):
        seed = (seeds[i] if seeds else (1234 + i * 7))
        _set_seed(seed)
        ids = tokenizer.encode_ids(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        max_seq_len = int(model.embedding.max_seq_len)
        eos_token_id = tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            x = input_ids[:, -max_seq_len:]
            with torch.inference_mode():
                logits = model(x)
            nxt = _sample_next_token(
                logits=logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_ids=input_ids[0],
                repetition_penalty=penalty,
            )
            input_ids = torch.cat([input_ids, nxt.unsqueeze(1)], dim=1)
            if eos_token_id is not None and int(nxt.item()) == int(eos_token_id):
                break
        # Score only the continuation, not the prompt copied into the decoded text.
        outputs.append(tokenizer.decode(input_ids[0, len(ids):].tolist()))
    return outputs


# ============================================================
# Metric: Diversity
# ============================================================
def distinct_n(texts, n=1):
    """Distinct-n ratio across combined generated tokens."""
    all_tokens = tokenize_for_metrics(" ".join(texts))
    if not all_tokens:
        return 0.0
    if n == 1:
        return len(set(all_tokens)) / len(all_tokens)
    ngrams = []
    tokens = all_tokens
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def repetition_ratio(texts):
    """Average within-output repetitions beyond each word's first use."""
    scores = []
    for text in texts:
        words = tokenize_for_metrics(text)
        if words:
            counts = Counter(words)
            scores.append(sum(count - 1 for count in counts.values()) / len(words))
    return sum(scores) / len(scores) if scores else 0.0


def unique_ratio_between(texts):
    """Proporsi output yang unik antar-generasi (diversity antar sample)."""
    if not texts:
        return 0.0
    norm = [t.strip() for t in texts]
    uniq = len(set(norm))
    return uniq / len(norm)


# ============================================================
# Metric: Fluency
# ============================================================
def average_length(texts):
    if not texts:
        return 0.0
    return sum(len(tokenize_for_metrics(t)) for t in texts) / len(texts)


def unk_ratio(tokenizer, texts):
    """Proporsi token UNK di dalam output (semakin rendah semakin baik)."""
    total = 0
    unk = 0
    unk_id = tokenizer.token2id.get(tokenizer.unk_token)
    for text in texts:
        ids = tokenizer.encode_ids(text)
        total += len(ids)
        unk += sum(1 for i in ids if i == unk_id)
    if total == 0:
        return 0.0
    return unk / total


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="development")
    parser.add_argument("--checkpoint", default="reports/checkpoints/gpt_last.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--out", default="reports/generation_quality_report.md")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    env = args.env
    cfg = get_config(env)
    device = _resolve_device(args.device)

    payload = _load_checkpoint_payload(checkpoint)
    ckpt_env = _checkpoint_env(payload)
    if ckpt_env and ckpt_env != env:
        env = ckpt_env
        cfg = get_config(env)
        print(f"[info] checkpoint env: {env}")

    # Load the tokenizer & build model matching the checkpoint's actual
    # architecture (read from its metadata), since the checkpoint uses
    # different dims than the current DevelopmentConfig.
    metadata_model = (payload.get("metadata") or {}).get("model") or {}
    ckpt_vocab = int(metadata_model.get("vocab_size", cfg.VOCAB_SIZE))
    ckpt_d_model = int(metadata_model.get("d_model", cfg.EMBEDDING_DIM))
    ckpt_layers = int(metadata_model.get("num_layers", cfg.TRANSFORMER_NUM_LAYERS))
    ckpt_max_seq = int(metadata_model.get("max_seq_len", cfg.EMBEDDING_MAX_SEQ_LEN))
    ckpt_heads = int(metadata_model.get("attention_num_heads", cfg.ATTENTION_NUM_HEADS))
    ckpt_kv_heads = int(
        (payload.get("metadata") or {}).get("model", {}).get("attention_num_kv_heads", ckpt_heads)
    )
    ckpt_pos = str(metadata_model.get("positional_embedding_type", "learned"))

    tokenizer, embedding = build_tokenizer_and_embedding(cfg=cfg, load_tokenizer=True)
    model = GPTModel.from_config(
        cfg=cfg,
        vocab_size=ckpt_vocab,
        embedding_layer=embedding,
    )
    # Override geometry to the checkpoint's values (identically to how it was trained).
    model.d_model = ckpt_d_model
    model.transformer.num_layers = ckpt_layers
    model.embedding.max_seq_len = ckpt_max_seq
    # Load the state dict.
    model_state = _resolve_model_state(payload)
    model.load_compatible_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()

    print("Loaded model+tester ...")
    print(f"  env         = {env}")
    print(f"  device      = {device}")
    print(f"  vocab_size  = {model.vocab_size}")
    print(f"  d_model     = {model.d_model}")
    print(f"  num_layers  = {model.transformer.num_layers}")
    print(f"  max_seq_len = {model.embedding.max_seq_len}")

    prompts = [
        "pengembangan model bahasa",
        "natural language processing",
        "tokenizer bahasa indonesia",
        "machine learning model",
        "sistem pembelajaran mesin",
    ]

    results = {}
    for p in prompts:
        outputs = generate_list(
            model, tokenizer, p, n=args.samples, device=device
        )
        results[p] = outputs

    # ============================
    # Compute metrics
    # ============================
    report = []
    report.append("# Evaluasi Generasi Checkpoint")
    report.append("")
    report.append(f"- **Environment**: `{env}`")
    report.append(f"- **Checkpoint**: `{args.checkpoint}`")
    report.append(f"- **Epoch / step**: `{payload.get('epoch', 'n/a')} / {payload.get('step', 'n/a')}`")
    report.append(f"- **Device**: `{device}`")
    report.append(f"- **Jumlah sampel per prompt**: `{args.samples}`")
    report.append(
        f"- **Decoding**: temperature `{TEMPERATURE}`, top-k `{TOP_K}`, top-p `{TOP_P}`, "
        f"repetition penalty `{REPETITION_PENALTY}`, maksimum `{MAX_NEW_TOKENS}` token baru"
    )
    report.append("")

    all_outputs = [out for outs in results.values() for out in outs]

    # Config metrics per prompt + overall
    global_metrics = {}
    global_metrics["distinct_1"] = distinct_n(all_outputs, 1)
    global_metrics["distinct_2"] = distinct_n(all_outputs, 2)
    global_metrics["repetition_ratio"] = repetition_ratio(all_outputs)
    global_metrics["unique_between"] = unique_ratio_between(all_outputs)
    global_metrics["avg_length_words"] = average_length(all_outputs)
    global_metrics["unk_ratio"] = unk_ratio(tokenizer, all_outputs)

    # Per-prompt diversity
    per_prompt = {}
    for p, outputs in results.items():
        per_prompt[p] = {
            "distinct_1": distinct_n(outputs, 1),
            "distinct_2": distinct_n(outputs, 2),
            "unique": unique_ratio_between(outputs),
            "repetition_ratio": repetition_ratio(outputs),
        }

    # ============================
    # Build markdown report
    # ============================
    report.append("## Hasil Generasi (sampel)")
    report.append("")
    for p, outputs in results.items():
        report.append(f"### Prompt: `{p}`")
        report.append("")
        for i, out in enumerate(outputs, 1):
            display = out if out.strip() else "*(berhenti langsung; tidak menghasilkan token lanjutan)*"
            report.append(f"**Sample {i}**: {display}")
            report.append("")
        report.append("")

    report.append("## Skor Kualitas Per Prompt")
    report.append("")
    report.append("| Prompt | Distinct-1 | Distinct-2 | Unique | Repetition |")
    report.append("|--------|-----------|-----------|--------|-----------|")
    for p, m in per_prompt.items():
        report.append(
            f"| `{p}` | {m['distinct_1']:.3f} | {m['distinct_2']:.3f} | "
            f"{m['unique']:.2f} | {m['repetition_ratio']:.3f} |"
        )
    report.append("")

    report.append("## Metrik Leksikal (Semua Output)")
    report.append("")
    report.append("| Metrik | Nilai | Catatan |")
    report.append("|--------|-------|---------|")
    report.append(
        f"| **Diversity (Distinct-1)** | {global_metrics['distinct_1']:.3f} | "
        f"Rasio kata unik terhadap seluruh kata; bukan skor semantik |"
    )
    report.append(
        f"| **Diversity (Distinct-2)** | {global_metrics['distinct_2']:.3f} | "
        f"Rasio bigram unik; bukan skor koherensi |"
    )
    report.append(
        f"| **Repetition Ratio** | {global_metrics['repetition_ratio']:.3f} | "
        f"Pengulangan dalam tiap output (rendah berarti lebih sedikit) |"
    )
    report.append(
        f"| **Unique antara sampel** | {global_metrics['unique_between']:.2f} | "
        f"Diversity output antar-generasi |"
    )
    report.append(
        f"| **Rata-rata panjang (kata)** | {global_metrics['avg_length_words']:.1f} | "
        f"Jumlah kata per output |"
    )
    report.append("")

    report.append("## Batas Evaluasi")
    report.append("")
    report.append(
        "- Metrik leksikal tidak mengukur relevansi prompt, tata bahasa, atau koherensi."
    )
    report.append(
        "- Penilaian semantik masih perlu dilakukan manusia pada prompt dan kriteria "
        "yang ditetapkan; beberapa sampel ini bukan evaluasi menyeluruh."
    )
    report.append("")

    md = "\n".join(report)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "=" * 70)
    print("SKOR KUALITAS GLOBAL")
    print("=" * 70)
    for k, v in global_metrics.items():
        print(f"  {k:28s} = {v:.4f}")
    print("\nRincian per prompt:")
    for p, m in per_prompt.items():
        print(f"  Prompt '{p}':")
        for k, v in m.items():
            print(f"    {k:20s} = {v:.4f}")
    print(f"\nReport disimpan ke: {out_path}")

    # Save JSON for programmatic use.
    json_out = out_path.with_suffix(".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(
            {"global": global_metrics, "per_prompt": per_prompt, "outputs": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"JSON disimpan ke: {json_out}")


if __name__ == "__main__":
    main()

