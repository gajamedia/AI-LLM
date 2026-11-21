"""
Menyatukan dataset instruksi bahasa Indonesia (data/hf_sft) menjadi satu JSONL SFT.

Keluaran per baris: {"prompt", "response", "source", "n_tokens"}.
Prioritas sumber saat instruksi duplikat: alpaca_gpt4 > oasst_top1 > sharegpt > alpaca_cahya.
Contoh yang melebihi --max-tokens (prompt+jawaban dengan template) dibuang, bukan dipotong,
karena jawaban terpotong mengajarkan model berhenti di tengah kalimat.
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))

TEMPLATE_PROMPT = "Pertanyaan: {p}\nJawaban:"


def norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_alpaca_gpt4(path):
    csv.field_size_limit(10 ** 9)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        yield r["input"].strip(), r["output"].strip(), "alpaca_gpt4"


def load_oasst(path):
    csv.field_size_limit(10 ** 9)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        yield r["instruction_id"].strip(), r["output_id"].strip(), "oasst_top1"


def load_sharegpt(path):
    # hanya giliran pertama (human -> gpt); percakapan multi-giliran tidak dipakai di tahap ini
    for d in json.load(open(path, encoding="utf-8")):
        c = d.get("conversations") or []
        if len(c) >= 2 and c[0]["from"] == "human" and c[1]["from"] == "gpt":
            yield c[0]["value"].strip(), c[1]["value"].strip(), "sharegpt"


def load_cahya(path):
    for line in open(path, encoding="utf-8"):
        t = json.loads(line)["text"]
        m = re.search(r"### Instruction:\n(.*?)\n\n### (?:Input:\n(.*?)\n\n### )?Response:\n(.*)", t, re.S)
        if not m:
            continue
        instr, inp, out = m.group(1).strip(), (m.group(2) or "").strip(), m.group(3).strip()
        prompt = f"{instr}\n{inp}" if inp else instr
        yield prompt, out, "alpaca_cahya"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="data/hf_sft")
    ap.add_argument("--out", default="data/sft/sft_id.jsonl")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--min-response-chars", type=int, default=15)
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from config import get_config
    from embedding import build_tokenizer_and_embedding
    cfg = get_config("production_100m")
    tokenizer, _ = build_tokenizer_and_embedding(cfg=cfg, load_tokenizer=True)

    src = Path(args.src_dir)
    loaders = [
        load_alpaca_gpt4(src / "alpaca_gpt4_id.csv"),
        load_oasst(src / "oasst_top1_id.csv"),
        load_sharegpt(src / "sharegpt_id.json"),
        load_cahya(src / "alpaca_id_cahya.jsonl"),
    ]
    seen, kept, stats = set(), [], Counter()
    for loader in loaders:
        for prompt, resp, name in loader:
            stats[f"{name}:read"] += 1
            if len(resp) < args.min_response_chars or not prompt:
                stats[f"{name}:too_short"] += 1
                continue
            key = norm(prompt)
            if key in seen:
                stats[f"{name}:duplicate"] += 1
                continue
            full = TEMPLATE_PROMPT.format(p=prompt) + " " + resp
            n = len(tokenizer.encode_ids(full, add_bos=True, add_eos=True))
            if n > args.max_tokens:
                stats[f"{name}:too_long"] += 1
                continue
            seen.add(key)
            stats[f"{name}:kept"] += 1
            kept.append({"prompt": prompt, "response": resp, "source": name, "n_tokens": n})

    random.Random(args.seed).shuffle(kept)
    val, train = kept[:args.val_size], kept[args.val_size:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((out, train), (out.with_name(out.stem + "_val.jsonl"), val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k in sorted(stats):
        print(f"{k}: {stats[k]}")
    toks = sorted(r["n_tokens"] for r in kept)
    print(f"total kept={len(kept)} train={len(train)} val={len(val)}")
    print(f"tokens/contoh: median={toks[len(toks)//2]} p90={toks[int(len(toks)*0.9)]} total={sum(toks):,}")
    print("saved:", out)


if __name__ == "__main__":
    main()
