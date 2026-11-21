"""
SFT tahap 2: QA berbasis teks (jawab dari konteks, atau katakan tidak ada).

Sumber: TyDi QA (secondary_task, bahasa Indonesia; Apache-2.0) dan IDK-MRC (CC-BY 4.0; memuat
pertanyaan tak terjawab). Konteks dipotong ke jendela kalimat di sekitar jawaban agar muat dalam
256 token. Ditambah sampel SFT umum (data/sft/sft_id.jsonl) agar kemampuan instruksi tidak hilang.

Keluaran:
  data/sft/sft_qa.jsonl       latih (QA + campuran umum)
  data/sft/sft_qa_val.jsonl   validasi (QA held-out + sedikit umum)
  data/sft/sft_qa_eval.jsonl  set uji QA held-out dengan kunci jawaban (answerable / jawaban emas)
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))

TEMPLATE_PROMPT = "Pertanyaan: {p}\nJawaban:"
QA_PROMPT = "Bacalah teks berikut.\n{ctx}\nBerdasarkan teks tersebut, jawab: {q}"
NO_ANSWER = [
    "Informasi tersebut tidak ada dalam teks.",
    "Teks tidak menyebutkan hal itu.",
    "Maaf, jawabannya tidak terdapat dalam teks yang diberikan.",
]
BAD = "�"
MAX_CTX_CHARS = 600


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def window(context, answer_start, answer_len, budget=MAX_CTX_CHARS):
    """Jendela kalimat berurutan di sekitar jawaban (atau dari awal bila answer_start None)."""
    sents, pos = [], 0
    for s in sentences(context):
        i = context.find(s, pos)
        sents.append((i, i + len(s), s))
        pos = i + len(s)
    if not sents:
        return context[:budget]
    if answer_start is None:
        center = 0
    else:
        center = next((k for k, (a, b, _) in enumerate(sents) if a <= answer_start < b), 0)
    lo = hi = center
    size = len(sents[center][2])
    while True:
        grew = False
        if hi + 1 < len(sents) and size + len(sents[hi + 1][2]) + 1 <= budget:
            hi += 1; size += len(sents[hi][2]) + 1; grew = True
        if lo - 1 >= 0 and size + len(sents[lo - 1][2]) + 1 <= budget:
            lo -= 1; size += len(sents[lo][2]) + 1; grew = True
        if not grew:
            break
    out = " ".join(s for _, _, s in sents[lo:hi + 1])
    return out[:budget] if size > budget else out  # kalimat tunggal yang kepanjangan


def iter_tydi(path):
    df = pd.read_parquet(path)
    df = df[df["id"].str.startswith("indonesian")]
    for r in df.itertuples():
        ans = r.answers["text"][0]
        yield "tydi", r.context, r.question, ans, int(r.answers["answer_start"][0])


def iter_idk(path):
    for c in json.load(open(path, encoding="utf-8")):
        for q in c["qas"]:
            if q.get("is_impossible"):
                yield "idkmrc", c["context"], q["question"], None, None
            else:
                a = q["answers"][0]
                yield "idkmrc", c["context"], q["question"], a["text"], int(a["answer_start"])


def build(items, tokenizer, rng, max_tokens, stats, tag):
    rows = []
    for src, ctx, q, ans, start in items:
        stats[f"{tag}:{src}:read"] += 1
        ctx, q = clean(ctx) if ans is None else ctx, clean(q)
        if BAD in ctx or BAD in q or (ans and BAD in ans):
            stats[f"{tag}:{src}:encoding_rusak"] += 1
            continue
        if ans is None:
            w = clean(window(ctx, None, 0))
            resp = rng.choice(NO_ANSWER)
        else:
            w = clean(window(ctx, start, len(ans)))
            ans = clean(ans)
            if ans not in w:
                stats[f"{tag}:{src}:jawaban_hilang_dari_jendela"] += 1
                continue
            resp = ans
        prompt = QA_PROMPT.format(ctx=w, q=q)
        n = len(tokenizer.encode_ids(TEMPLATE_PROMPT.format(p=prompt) + " " + resp, add_bos=True, add_eos=True))
        if n > max_tokens:
            stats[f"{tag}:{src}:too_long"] += 1
            continue
        stats[f"{tag}:{src}:kept"] += 1
        rows.append({"prompt": prompt, "response": resp, "source": f"qa_{src}", "n_tokens": n,
                     "answerable": ans is not None, "question": q, "context": w})
    return rows


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="data/hf_sft")
    ap.add_argument("--general", default="data/sft/sft_id.jsonl")
    ap.add_argument("--general-val", default="data/sft/sft_id_val.jsonl")
    ap.add_argument("--general-n", type=int, default=15000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--eval-size", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/sft")
    args = ap.parse_args()

    from config import get_config
    from embedding import build_tokenizer_and_embedding
    tokenizer, _ = build_tokenizer_and_embedding(cfg=get_config("production_100m"), load_tokenizer=True)
    rng = random.Random(args.seed)
    src, stats = Path(args.src_dir), Counter()

    train = build(list(iter_tydi(src / "tydiqa_secondary_train.parquet")) +
                  list(iter_idk(src / "idkmrc_train.json")), tokenizer, rng, args.max_tokens, stats, "train")
    heldout = build(list(iter_tydi(src / "tydiqa_secondary_valid.parquet")) +
                    list(iter_idk(src / "idkmrc_valid.json")), tokenizer, rng, args.max_tokens, stats, "heldout")

    rng.shuffle(heldout)
    eval_set = heldout[:args.eval_size]
    qa_val = heldout[args.eval_size:]

    general = [json.loads(l) for l in open(args.general, encoding="utf-8")]
    rng.shuffle(general)
    general_train = general[:args.general_n]
    general_val = [json.loads(l) for l in open(args.general_val, encoding="utf-8")][:300]

    full_train = train + general_train
    rng.shuffle(full_train)
    val = qa_val + general_val
    out = Path(args.out_dir)
    write(out / "sft_qa.jsonl", full_train)
    write(out / "sft_qa_val.jsonl", val)
    write(out / "sft_qa_eval.jsonl", eval_set)

    for k in sorted(stats):
        print(f"{k}: {stats[k]}")
    ans = sum(r["answerable"] for r in train)
    print(f"QA latih={len(train)} (terjawab={ans}, tak terjawab={len(train)-ans}) + umum={len(general_train)} -> {len(full_train)}")
    print(f"validasi={len(val)} (QA={len(qa_val)}, umum={len(general_val)}) | uji held-out={len(eval_set)} "
          f"(terjawab={sum(r['answerable'] for r in eval_set)})")


if __name__ == "__main__":
    main()
