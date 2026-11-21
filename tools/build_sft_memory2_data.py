"""
SFT tahap 5: catatan rumit kompositional (memory_gen2) + sebagian data memori v1 + replay.
Set uji tulisan tangan (data/sft/memory_complex*.jsonl) TIDAK dipakai dan tidak boleh disalin ke sini.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_gen2 as G2
import rag_lib as R


def to_row(ex, rng):
    resp = ex["answer"] if ex["answer"] is not None else rng.choice(R.NO_ANSWER)
    return {"prompt": R.make_prompt(ex["ctx"], ex["question"]), "response": resp,
            "source": "mem2_" + ex["kind"], "question": ex["question"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-v2", type=int, default=14000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--mem-v1", default="data/sft/sft_mem.jsonl")
    ap.add_argument("--keep-v1-mem", type=int, default=5000)
    ap.add_argument("--keep-replay", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out-dir", default="data/sft")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    v2_train = [to_row(G2.make_example2(rng), rng) for _ in range(args.n_v2)]
    v2_val = [to_row(G2.make_example2(rng), rng) for _ in range(args.n_val)]
    old = [json.loads(l) for l in open(args.mem_v1, encoding="utf-8")]
    old_mem = [r for r in old if r["source"].startswith("mem_")]
    old_rep = [r for r in old if not r["source"].startswith("mem_")]
    rng.shuffle(old_mem); rng.shuffle(old_rep)
    train = v2_train + old_mem[:args.keep_v1_mem] + old_rep[:args.keep_replay]
    rng.shuffle(train)
    val = v2_val + old_mem[args.keep_v1_mem:args.keep_v1_mem + 200] + old_rep[args.keep_replay:args.keep_replay + 200]
    out = Path(args.out_dir)
    for path, rows in ((out / "sft_mem2.jsonl", train), (out / "sft_mem2_val.jsonl", val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("latih:", len(train), "| validasi:", len(val))
    print("jenis v2:", dict(Counter(r["source"] for r in v2_train)))
    seen = set()
    for r in v2_train:
        k = r["source"] + str(len(seen) % 3)
        if k in seen:
            continue
        seen.add(k)
        if len(seen) > 6:
            break
        print(f"\n[{r['source']}]\n{r['prompt'][22:]}\n=> {r['response']!r}")


if __name__ == "__main__":
    main()
