"""
Uji kalimat rumit tulisan tangan (set yang dibekukan, data/sft/memory_complex*.jsonl) untuk pembaca memori.
Memori = semua catatan rumit + 300 catatan sintetis pengecoh. Tiap soal dibaca per catatan top-3 (di-cache).
Kebijakan: oracle (catatan benar langsung), raw_top1, agen ambang 0,5 dan 0,67 (pagar pengaman).
"""
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_agent as M
import memory_gen as G
import rag_lib as R


def outcome(gold, ans):
    if ans is None:
        return "abstain"
    return "benar" if R.norm(gold) in R.norm(ans) else "salah"


def apply(q, reads):
    top1 = reads[0][2] if reads and not M.refusal(reads[0][2]) else None
    d5, d67 = M.decide(q, reads, True, True, 0.5), M.decide(q, reads, True, True, 0.67)
    return {"raw_top1": top1,
            "agen_0.5": d5["answer"] if d5["status"] != "unknown" else None,
            "agen_0.67": d67["answer"] if d67["status"] != "unknown" else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    reader = M.Reader(args.checkpoint)
    rows = [json.loads(l) for l in open("evalsets/memory_complex.jsonl", encoding="utf-8")]
    unans = [json.loads(l)["question"] for l in open("evalsets/memory_complex_unans.jsonl", encoding="utf-8")]
    rng = random.Random(args.seed)
    store = M.MemoryStore()
    for n in dict.fromkeys(r["note"] for r in rows):
        store.teach(n)
    for _ in range(300):
        store.teach(G.make_fact(rng, rng.choice(G.TRAIN_ATTRS))["note"])

    tally = {k: {"benar": 0, "salah": 0, "abstain": 0} for k in ("oracle", "raw_top1", "agen_0.5", "agen_0.67")}
    un = {k: {"abstain": 0, "mengarang": 0} for k in ("raw_top1", "agen_0.5", "agen_0.67")}
    recall3, cache = 0, []

    for r in rows:
        hits = store.search(r["question"], 3)
        recall3 += int(any(h["text"] == r["note"] for h in hits))
        reads = [(h["id"], h["text"], reader.read(h["text"], r["question"])) for h in hits]
        orc = reader.read(r["note"], r["question"])
        outs = {"oracle": None if M.refusal(orc) else orc, **apply(r["question"], reads)}
        for k, a in outs.items():
            tally[k][outcome(r["answer"], a)] += 1
        cache.append({"q": r["question"], "gold": r["answer"], "outs": outs})
    for q in unans:
        hits = store.search(q, 3)
        reads = [(h["id"], h["text"], reader.read(h["text"], q)) for h in hits]
        outs = apply(q, reads)
        for k, a in outs.items():
            un[k]["abstain" if a is None else "mengarang"] += 1
        cache.append({"q": q, "gold": None, "outs": outs})
    n, nu = len(rows), len(unans)
    rep = {"n_terjawab": n, "n_tak_terjawab": nu, "recall@3": recall3 / n,
           "terjawab": {k: {x: v / n for x, v in t.items()} for k, t in tally.items()},
           "tak_terjawab": {k: {x: v / nu for x, v in t.items()} for k, t in un.items()}, "cache": cache}
    print(f"n terjawab={n}, tak terjawab={nu}, recall@3={recall3/n:.2f}")
    print(f"{'kebijakan':12s}{'benar':>8s}{'salah':>8s}{'abstain':>9s}")
    for k, t in rep["terjawab"].items():
        print(f"{k:12s}{t['benar']:8.2f}{t['salah']:8.2f}{t['abstain']:9.2f}")
    print(f"{'tak terjawab':14s}{'abstain':>8s}{'mengarang':>11s}")
    for k, t in rep["tak_terjawab"].items():
        print(f"{k:14s}{t['abstain']:8.2f}{t['mengarang']:11.2f}")
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
