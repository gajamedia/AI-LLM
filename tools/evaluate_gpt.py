"""
Evaluasi formal checkpoint GPT: loss/perplexity pada token validasi + sampel generasi.

Loss dihitung pada jendela acak 257 token dari aliran token validasi (seed tetap),
sehingga dua checkpoint dengan tokenizer yang sama dinilai pada data identik.
Catatan: jendela memotong dokumen sembarang, jadi angkanya tidak persis sama dengan
validation_loss saat training (yang memakai blok packed); bandingkan antar-checkpoint
dengan skrip yang sama, bukan dengan log training.
"""
import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from config import get_config
from generate_gpt import _load_checkpoint_payload, _load_model, _set_seed, generate_text

PROMPTS = [
    "Indonesia adalah negara",
    "Cara membuat nasi goreng yang enak adalah",
    "Pemerintah Indonesia hari ini mengumumkan bahwa",
    "Sejarah kerajaan Majapahit dimulai pada",
    "Ibu kota Indonesia adalah",
    "Manfaat olahraga secara teratur bagi kesehatan antara lain",
    "Pada pertandingan sepak bola semalam, tim nasional",
    "Untuk mengajukan permohonan kartu tanda penduduk, warga harus",
    "Fotosintesis adalah proses",
    "Harga kebutuhan pokok di pasar tradisional minggu ini",
    "Dalam bahasa pemrograman Python, fungsi digunakan untuk",
    "Banjir yang melanda kota tersebut menyebabkan",
]


def words(text):
    return re.findall(r"[a-z0-9À-ɏ]+", text.lower())


def distinct_n(texts, n):
    toks = words(" ".join(texts))
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_ratio(texts):
    scores = []
    for t in texts:
        w = words(t)
        if w:
            scores.append(sum(c - 1 for c in Counter(w).values()) / len(w))
    return sum(scores) / len(scores) if scores else 0.0


def eval_loss(model, token_file, blocks, seq_len, batch, device, seed):
    data = np.memmap(token_file, dtype=np.int32, mode="r")
    n_windows = (len(data) - 1) // seq_len
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_windows, size=min(blocks, n_windows), replace=False)
    total_loss, total_tok = 0.0, 0
    t0 = time.time()
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        arr = np.stack([data[i * seq_len:i * seq_len + seq_len + 1] for i in chunk]).astype(np.int64)
        x = torch.from_numpy(arr[:, :-1]).to(device)
        y = torch.from_numpy(arr[:, 1:]).to(device)
        with torch.inference_mode():
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum")
        total_loss += float(loss)
        total_tok += y.numel()
        if (s // batch) % 20 == 0:
            print(f"  loss {s + len(chunk)}/{len(idx)} blok, {time.time() - t0:.0f}s", flush=True)
    mean = total_loss / total_tok
    return {"loss": mean, "ppl": math.exp(mean), "tokens": total_tok, "blocks": len(idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="production_100m")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-token-file", default="data/prod_v2/validation_tokens.bin")
    ap.add_argument("--blocks", type=int, default=3000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gen-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device(args.device)
    cfg = get_config(args.env)
    ckpt = Path(args.checkpoint)
    payload = _load_checkpoint_payload(ckpt)
    step = payload.get("step")
    model, tokenizer = _load_model(ckpt, cfg, device, payload=payload)
    del payload

    print(f"checkpoint={ckpt} step={step}")
    res = {"checkpoint": str(ckpt), "step": step}
    res["validation"] = eval_loss(model, args.val_token_file, args.blocks, args.seq_len, args.batch, device, args.seed)
    print("validation:", res["validation"], flush=True)

    outputs = {}
    for p in PROMPTS:
        outs = []
        for k in range(args.gen_samples):
            _set_seed(args.seed + k)
            text = generate_text(model, tokenizer, p, args.max_new_tokens, 0.8, 40, 0.95, 1.1, device)
            outs.append(text[len(p):] if text.startswith(p) else text)
        outputs[p] = outs
        print(f"[gen] {p} -> {outs[0][:100]!r}", flush=True)
    flat = [o for outs in outputs.values() for o in outs]
    res["generation"] = {
        "distinct_1": distinct_n(flat, 1),
        "distinct_2": distinct_n(flat, 2),
        "distinct_3": distinct_n(flat, 3),
        "repetition_ratio": repetition_ratio(flat),
        "avg_words": sum(len(words(o)) for o in flat) / len(flat),
        "samples": len(flat),
    }
    res["outputs"] = outputs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("generation:", res["generation"])
    print("saved:", args.out)


if __name__ == "__main__":
    main()
