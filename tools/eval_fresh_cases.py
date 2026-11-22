"""
Ujian kasus segar untuk satu atau beberapa checkpoint (dipakai sebelum dan sesudah SFT v7).
  python tools/eval_fresh_cases.py reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt reports/checkpoints_100m_sft_mem4/gpt_sft_best.pt
Kasus di evalsets/memory_cases_v7_segar.jsonl memakai entitas dan kalimat yang TIDAK ada di pembangkit data (dijaga tes).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "apps"))

import memory_agent as M
import model_acceptance_eval as E


def run(checkpoint, path=PROJECT_ROOT / "evalsets" / "memory_cases_v7_segar.jsonl"):
    cases = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    reader = M.Reader(checkpoint)
    rows = []
    for c in cases:
        res = E.eval_case(E._case_agent(c, reader), c)
        rows.append({"id": c["id"], **res})
    return rows


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:])
    for p in paths:
        rows = run(p)
        ok = sum(r["lolos"] for r in rows)
        print(f"\n{p}\n  lolos {ok}/{len(rows)}")
        for r in rows:
            print(f"  {'OK ' if r['lolos'] else 'GAGAL'} {r['id']:22s} status={r['status']:8s} jawaban={r.get('jawaban')!r}")


if __name__ == "__main__":
    main()
