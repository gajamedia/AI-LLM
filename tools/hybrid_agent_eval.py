"""
Ukur AGEN UTUH (model 100M sungguhan) pada set kasus beku evalsets/memory_cases*.jsonl, dengan dan tanpa pencarian hibrida
(BM25 + E5) dan dengan/tanpa cadangan deterministik 'kategori mana/apa'. Dua susunan memori:
  per-kasus : tiap kasus hanya berisi catatannya sendiri (seperti set aslinya)
  gabungan  : semua catatan semua kasus dalam SATU memori (pencarian harus memilih catatan yang benar)
Ditambah pertanyaan pengguna sendiri: "Tim mana yang menggunakan ruangan anggrek".
  python tools/hybrid_agent_eval.py --embed intfloat/multilingual-e5-small
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hybrid_search as H
import memory_agent as M
import rag_lib as R

PROBE = {"notes": ["Rapat tim produk dimulai pukul 09.30 di ruang Anggrek.", "Rapat tim finance dimulai pukul 11.30 di ruang Mawar."],
         "question": "Tim mana yang menggunakan ruangan anggrek", "answer": "produk"}


def load(name):
    return [json.loads(l) for l in open(ROOT / "evalsets" / name, encoding="utf-8") if l.strip()]


def make_store(notes, embedder):
    s = M.MemoryStore()
    for n in dict.fromkeys(notes):
        s.teach(n)
    if embedder is not None:
        s._hv_tried, s._hv = True, H.NoteVectors(s.con, embedder)
    return s


def grade(o, gold):
    a = o["answer"] if o["status"] in ("ok", "conflict") else None
    if a is None:
        return "belum_tahu", None
    return ("benar" if R.norm(gold) in R.norm(a) else "salah"), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default=H.DEFAULT_EMBED_MODEL)
    ap.add_argument("--out", default="reports/eval_hybrid_agent.json")
    args = ap.parse_args()
    reader = M.Reader(str(ROOT / "reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt"))
    emb = H.E5Embedder(args.embed)
    cases = load("memory_cases.jsonl") + load("memory_cases_v7_segar.jsonl")
    cases = [c for c in cases if c.get("answer")]
    all_notes = [n for c in cases for n in c["notes"]]
    res = {}
    for layout in ("per-kasus", "gabungan"):
        for search in ("BM25", "hibrida"):
            for fb in (False, True):
                e = emb if search == "hibrida" else None
                shared = make_store(all_notes, e) if layout == "gabungan" else None
                tally = {"benar": 0, "salah": 0, "belum_tahu": 0}
                detail = []
                for c in cases + [PROBE]:
                    store = shared or make_store(c["notes"], e)
                    agent = M.MemoryAgent(store, reader)
                    agent.category_fallback = fb
                    o = agent.ask(c["question"])
                    g, a = grade(o, c["answer"])
                    if c is not PROBE:
                        tally[g] += 1
                    detail.append({"q": c["question"], "gold": c["answer"], "hasil": g, "jawaban": a, "via": o.get("via")})
                key = f"{layout} | {search} | cadangan {'ON' if fb else 'OFF'}"
                probe = detail[-1]
                res[key] = {"n": len(cases), **tally, "probe": probe, "detail": detail}
                print(f"{key:<44} benar={tally['benar']}/{len(cases)} salah={tally['salah']} belum_tahu={tally['belum_tahu']} "
                      f"| probe: {probe['hasil']} ({probe['jawaban']})", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
