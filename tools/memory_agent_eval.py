"""
Ukur efek pagar pengaman pada pembaca memori. Tiap soal dibaca sekali per catatan top-3 (di-cache),
lalu empat kebijakan diterapkan pada keluaran yang sama:
  raw_top1      : jawaban dari catatan peringkat 1 apa adanya (kecuali penolakan)
  seq_nonrefuse : jawaban pertama yang bukan penolakan dari top-3
  verified      : + jawaban harus potongan teks catatan
  verified+ent  : + kata bernama/angka di pertanyaan harus ada di catatan
  verified+ent+rel : + akar kata inti pertanyaan harus ada di catatan (kebijakan aplikasi)
Soal terjawab: benar / salah (menjawab tapi keliru) / abstain.  Soal tak terjawab: abstain (baik) / mengarang (buruk).
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
import memory_test as T
import rag_lib as R


def policies(question, reads):
    first_non = next(((n, a) for _, n, a in reads if not M.refusal(a)), None)
    top1 = reads[0][2] if reads and not M.refusal(reads[0][2]) else None
    return {
        "raw_top1": top1,
        "seq_nonrefuse": first_non[1] if first_non else None,
        "verified": (lambda d: d["answer"] if d["status"] in ("ok", "conflict") else None)(M.decide(question, reads, False, False)),
        "verified+ent": (lambda d: d["answer"] if d["status"] in ("ok", "conflict") else None)(M.decide(question, reads, True, False)),
        "verified+ent+rel": (lambda d: d["answer"] if d["status"] in ("ok", "conflict") else None)(M.decide(question, reads, True, True)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt")
    ap.add_argument("--sizes", default="200,2000")
    ap.add_argument("--n-answerable", type=int, default=100)
    ap.add_argument("--n-unanswerable", type=int, default=50)
    ap.add_argument("--out", default="reports/eval_memory_agent.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    reader = M.Reader(args.checkpoint)
    suites = {}

    # --- tulisan tangan ---
    hw = [json.loads(l) for l in open("evalsets/memory_handwritten.jsonl", encoding="utf-8")]
    rng = random.Random(args.seed)
    store = M.MemoryStore()
    for n in dict.fromkeys(r["note"] for r in hw):
        store.teach(n)
    for _ in range(300):
        store.teach(G.make_fact(rng, rng.choice(G.TRAIN_ATTRS))["note"])
    from memory_eval2 import UNANSWERABLE_HW
    items = [(r["question"], r["answer"]) for r in hw] + [(q, None) for q in UNANSWERABLE_HW]
    suites["handwritten"] = (store, items)

    # --- sintetis (fakta lama, pola templat berbeda dari data latih) ---
    for size in [int(s) for s in args.sizes.split(",")]:
        rng = random.Random(args.seed + size)
        facts, persons = T.make_facts(rng, size)
        store = M.MemoryStore()
        for f in facts:
            store.teach(f[0])
        ans = [(f[1], f[2]) for f in rng.sample(facts, min(args.n_answerable, len(facts)))]
        un = []
        for _ in range(args.n_unanswerable):
            known = rng.choice(persons)
            fake = f"{known.split()[0]} {T.word(rng, 2).capitalize()}{T.word(rng, 1)}"
            un.append((rng.choice([f"Di kota mana {fake} dilahirkan?", f"Apa pekerjaan {fake}?",
                                   f"Apa makanan kesukaan {fake}?", f"Di lantai berapa ruang kerja {fake}?"]), None))
        suites[f"sintetis_{size}"] = (store, ans + un)

    report = {}
    for name, (store, items) in suites.items():
        tally = {p: {"benar": 0, "salah": 0, "abstain_ans": 0, "abstain_un": 0, "mengarang_un": 0} for p in
                 ("raw_top1", "seq_nonrefuse", "verified", "verified+ent", "verified+ent+rel")}
        n_ans = sum(1 for _, g in items if g is not None)
        n_un = len(items) - n_ans
        cache = []
        for q, gold in items:
            hits = store.search(q, 3)
            reads = [(h["id"], h["text"], reader.read(h["text"], q)) for h in hits]
            cache.append({"q": q, "gold": gold, "reads": reads})
            for p, ans in policies(q, reads).items():
                t = tally[p]
                if gold is not None:
                    if ans is None: t["abstain_ans"] += 1
                    elif R.norm(gold) in R.norm(ans): t["benar"] += 1
                    else: t["salah"] += 1
                else:
                    t["abstain_un" if ans is None else "mengarang_un"] += 1
        report[name] = {"n_terjawab": n_ans, "n_tak_terjawab": n_un, "kebijakan": tally, "reads": cache}
        print(f"\n== {name}: terjawab={n_ans}, tak terjawab={n_un}", flush=True)
        print(f"{'kebijakan':16s}{'benar':>8s}{'salah':>8s}{'abstain':>9s}{'| abstain(tak)':>15s}{'mengarang(tak)':>16s}", flush=True)
        for p, t in tally.items():
            print(f"{p:16s}{t['benar']/n_ans:8.2f}{t['salah']/n_ans:8.2f}{t['abstain_ans']/n_ans:9.2f}"
                  f"{t['abstain_un']/max(n_un,1):15.2f}{t['mengarang_un']/max(n_un,1):16.2f}", flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
