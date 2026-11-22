"""
Ukur efek 'bentuk pertanyaan alternatif' pada agen penuh (model sungguhan) di set uji beku:
bandingkan agen tanpa dan dengan bentuk alternatif. Benar / salah (menjawab keliru) / belum tahu; tak terjawab: abstain / mengarang.
  python tools/memory_forms_eval.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_agent as M
import rag_lib as R


def load(path):
    return [json.loads(l) for l in open(PROJECT_ROOT / path, encoding="utf-8") if l.strip()]


def main():
    reader = M.Reader(str(PROJECT_ROOT / "reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt"))
    from memory_eval2 import UNANSWERABLE_HW
    suites = {"tulisan tangan": (load("evalsets/memory_handwritten.jsonl"), list(UNANSWERABLE_HW)),
              "kalimat rumit": (load("evalsets/memory_complex.jsonl"), [r["question"] for r in load("evalsets/memory_complex_unans.jsonl")])}
    for name, (ans_items, unans) in suites.items():
        store = M.MemoryStore()
        for n in dict.fromkeys(r["note"] for r in ans_items):
            store.teach(n)
        agent = M.MemoryAgent(store, reader)
        print(f"\n== {name}: terjawab={len(ans_items)}, tak terjawab={len(unans)}", flush=True)
        for label, flag, echo, unread in (("tanpa bentuk alternatif", False, None, None), ("dengan bentuk alternatif", True, None, None),
                                          ("+ penjaga gema", True, 1.0, None), ("+ penjaga catatan tak terbaca", True, None, 0.2),
                                          ("+ keduanya (bawaan)", True, 1.0, 0.2)):
            agent.alternate_forms, agent.echo_threshold, agent.unread_margin = flag, echo, unread
            t = {"benar": 0, "salah": 0, "abstain": 0, "abstain_tak": 0, "mengarang_tak": 0}
            changed = []
            for r in ans_items:
                o = agent.ask(r["question"])
                a = o["answer"] if o["status"] in ("ok", "conflict") else None
                t["abstain" if a is None else "benar" if R.norm(r["answer"]) in R.norm(a) else "salah"] += 1
                if o.get("rewritten"):
                    changed.append((r["question"], o["rewritten"], a))
            for q in unans:
                o = agent.ask(q)
                t["abstain_tak" if o["status"] not in ("ok", "conflict") else "mengarang_tak"] += 1
            print(f"{label:30s} benar {t['benar']:3d}  salah {t['salah']:3d}  belum tahu {t['abstain']:3d} | tak terjawab: abstain {t['abstain_tak']:3d} mengarang {t['mengarang_tak']:3d}", flush=True)
            if label.startswith('dengan'):
                for c in changed:
                    print("   dibaca ulang:", c)


if __name__ == "__main__":
    main()
