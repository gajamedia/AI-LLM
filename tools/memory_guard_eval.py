"""
Ukur penjaga 'jawaban hanya menggemakan pertanyaan' pada set uji beku, dengan model sungguhan.
Tiap soal dibaca sekali per catatan top-3 (di-cache); kebijakan penjaga diterapkan pada keluaran yang sama.
  python tools/memory_guard_eval.py --checkpoint reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_agent as M
import rag_lib as R

VARIANTS = [("tanpa penjaga", None), ("gema >= 1.0", 1.0), ("gema >= 0.5", 0.5), ("gema > 0", 1e-9)]


def load(path):
    return [json.loads(l) for l in open(PROJECT_ROOT / path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports/eval_guard_echo.json"))
    args = ap.parse_args()
    reader = M.Reader(args.checkpoint)

    suites = {}
    hw = load("evalsets/memory_handwritten.jsonl")
    from memory_eval2 import UNANSWERABLE_HW
    suites["tulisan tangan"] = (hw, list(UNANSWERABLE_HW))
    cx = load("evalsets/memory_complex.jsonl")
    suites["kalimat rumit"] = (cx, [r["question"] for r in load("evalsets/memory_complex_unans.jsonl")])

    report = {}
    for name, (ans_items, unans) in suites.items():
        store = M.MemoryStore()
        for n in dict.fromkeys(r["note"] for r in ans_items):
            store.teach(n)
        items = [(r["question"], r["answer"]) for r in ans_items] + [(q, None) for q in unans]
        reads = {}
        for q, _ in items:
            hits = store.search(q, 3)
            reads[q] = [(h["id"], h["text"], reader.read(h["text"], q)) for h in hits]
        n_ans, n_un = len(ans_items), len(unans)
        print(f"\n== {name}: terjawab={n_ans}, tak terjawab={n_un}", flush=True)
        print(f"{'penjaga':16s}{'benar':>8s}{'salah':>8s}{'abstain':>9s}{'| abstain(tak)':>15s}{'mengarang(tak)':>16s}", flush=True)
        report[name] = {}
        for label, thr in VARIANTS:
            t = {"benar": 0, "salah": 0, "abstain": 0, "abstain_tak": 0, "mengarang_tak": 0}
            for q, gold in items:
                d = M.decide(q, reads[q], True, True, 0.67, frozenset(), echo_threshold=thr)
                a = d["answer"] if d["status"] in ("ok", "conflict") else None
                if gold is not None:
                    t["abstain" if a is None else "benar" if R.norm(gold) in R.norm(a) else "salah"] += 1
                else:
                    t["abstain_tak" if a is None else "mengarang_tak"] += 1
            report[name][label] = t
            print(f"{label:16s}{t['benar']/n_ans:8.2f}{t['salah']/n_ans:8.2f}{t['abstain']/n_ans:9.2f}"
                  f"{t['abstain_tak']/max(n_un,1):15.2f}{t['mengarang_tak']/max(n_un,1):16.2f}   "
                  f"(n: {t['benar']}/{t['salah']}/{t['abstain']})", flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
