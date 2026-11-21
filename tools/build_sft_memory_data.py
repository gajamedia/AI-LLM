"""
SFT tahap 4: pembaca catatan memori (data sintetis dari tools/memory_gen.py) + replay.
Hanya pola kalimat A dan atribut latih yang dipakai; pola B dan atribut uji tidak pernah masuk data latih.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_gen as G
import rag_lib as R


def to_row(ex, rng):
    resp = ex["answer"] if ex["answer"] is not None else rng.choice(R.NO_ANSWER)
    return {"prompt": R.make_prompt(ex["ctx"], ex["question"]), "response": resp,
            "source": "mem_" + ex["kind"], "question": ex["question"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=14000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--replay", default="data/sft/sft_rag.jsonl")
    ap.add_argument("--replay-n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out-dir", default="data/sft")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    mem_train = [to_row(G.make_example(rng, G.TRAIN_ATTRS, "A"), rng) for _ in range(args.n_train)]
    mem_val = [to_row(G.make_example(rng, G.TRAIN_ATTRS, "A"), rng) for _ in range(args.n_val)]
    replay = [json.loads(l) for l in open(args.replay, encoding="utf-8")]
    rng.shuffle(replay)
    replay_train, replay_val = replay[:args.replay_n], replay[args.replay_n:args.replay_n + 300]
    train = mem_train + replay_train
    rng.shuffle(train)
    val = mem_val + replay_val
    out = Path(args.out_dir)
    for path, rows in ((out / "sft_mem.jsonl", train), (out / "sft_mem_val.jsonl", val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("latih:", len(train), "(memori", len(mem_train), "+ replay", len(replay_train), ") | validasi:", len(val))
    print("jenis memori:", dict(Counter(r["source"] for r in mem_train)))
    for kind in ("mem_answerable", "mem_correction", "mem_unanswerable_person", "mem_unanswerable_attr"):
        r = next((x for x in mem_train if x["source"] == kind), None)
        if r:
            print(f"\n[{kind}]\n{r['prompt'][22:]}\n=> {r['response']!r}")


if __name__ == "__main__":
    main()
