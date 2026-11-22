"""
SFT tahap 6: jenis fakta baru (tim, divisi, peran, jam buka dan tutup, mulai dan selesai, anggota, ruang) dari tools/memory_gen3.py,
dicampur dengan data tahap 5 (catatan rumit) dan tahap 4 (memori v1) dan replay agar kemampuan lama tidak hilang.
Set uji (evalsets/*, data/sft/memory_complex*, data/sft/memory_handwritten*) TIDAK dipakai dan tidak boleh disalin ke sini.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_gen3 as G3
import rag_lib as R


def to_row(ex, rng):
    resp = ex["answer"] if ex["answer"] is not None else rng.choice(R.NO_ANSWER)
    return {"prompt": R.make_prompt(ex["ctx"], ex["question"]), "response": resp, "source": "mem3_" + ex["kind"], "question": ex["question"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-v3", type=int, default=12000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--mem2", default="data/sft/sft_mem2.jsonl")
    ap.add_argument("--keep-mem2", type=int, default=4000)
    ap.add_argument("--keep-mem1", type=int, default=2500)
    ap.add_argument("--keep-replay", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=6006)
    ap.add_argument("--out-dir", default="data/sft")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    v3 = [to_row(G3.make_example3(rng), rng) for _ in range(args.n_v3)]
    v3_val = [to_row(G3.make_example3(rng), rng) for _ in range(args.n_val)]
    old = [json.loads(l) for l in open(args.mem2, encoding="utf-8")]
    mem2 = [r for r in old if r["source"].startswith("mem2_")]
    mem1 = [r for r in old if r["source"].startswith("mem_")]
    rep = [r for r in old if not r["source"].startswith(("mem_", "mem2_"))]
    for xs in (mem2, mem1, rep):
        rng.shuffle(xs)
    k2, k1, kr = args.keep_mem2, args.keep_mem1, args.keep_replay
    train = v3 + mem2[:k2] + mem1[:k1] + rep[:kr]
    val = v3_val + mem2[k2:k2 + 150] + mem1[k1:k1 + 100] + rep[kr:kr + 150]
    rng.shuffle(train)
    out = Path(args.out_dir)
    for path, rows in ((out / "sft_mem3.jsonl", train), (out / "sft_mem3_val.jsonl", val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("latih:", len(train), "| validasi:", len(val), "| jenis v3:", dict(Counter(r["source"] for r in v3)))
    return train, val


if __name__ == "__main__":
    main()
