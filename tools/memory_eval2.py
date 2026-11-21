"""
Uji generalisasi pembaca memori:
  1) pola kalimat B (tidak dilatih) pada atribut yang dilatih
  2) atribut yang tak pernah dilatih (hobi, umur), pola A dan B
  3) kalimat bebas tulisan tangan (evalsets/memory_handwritten.jsonl) lewat pencarian BM25
     di antara 20 catatan tulisan tangan + 300 catatan sintetis pengecoh
Metrik: akurasi (jawaban emas ada di keluaran dan bukan penolakan) untuk soal terjawab;
        tingkat penolakan untuk soal tak terjawab.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_gen as G
import rag_lib as R
from config import get_config
from generate_gpt import _load_checkpoint_payload, _load_model, _set_seed, generate_text
from memory_test import ask, build_memory, refusal

UNANSWERABLE_HW = [
    "Siapa direktur utama koperasi desa?", "Berapa harga gula pasir di Toko Sumber Rejeki?",
    "Kapan Dokter Anindya berhenti praktik?", "Di mana Pak Slamet tinggal?",
    "Siapa juara festival layang-layang tahun lalu?", "Apa nama istri Pak Darmawan?",
    "Berapa jumlah karyawan bagian akuntansi?", "Jam berapa bus jurusan Surabaya berangkat?",
]


def correct(gold, out):
    return R.norm(gold) in R.norm(out) and not refusal(out)


def direct_block(model, tok, device, rng, attrs, split, n):
    stats = defaultdict(lambda: [0, 0])
    for _ in range(n):
        ex = G.make_example(rng, attrs, split)
        out = ask(model, tok, ex["ctx"], ex["question"], device)
        if ex["answer"] is None:
            stats[ex["kind"]][1] += 1
            stats[ex["kind"]][0] += int(refusal(out))
        else:
            stats[ex["kind"]][1] += 1
            stats[ex["kind"]][0] += int(correct(ex["answer"], out))
    return {k: {"benar": a, "n": b, "rasio": a / max(b, 1)} for k, (a, b) in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-direct", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    device = torch.device("cpu")
    cfg = get_config("production_100m")
    ckpt = Path(args.checkpoint)
    payload = _load_checkpoint_payload(ckpt)
    model, tok = _load_model(ckpt, cfg, device, payload=payload)
    del payload

    rep = {}
    rep["seen_attrs_phrasing_A"] = direct_block(model, tok, device, random.Random(args.seed), G.TRAIN_ATTRS, "A", args.n_direct)
    rep["seen_attrs_phrasing_B"] = direct_block(model, tok, device, random.Random(args.seed), G.TRAIN_ATTRS, "B", args.n_direct)
    rep["unseen_attrs_A"] = direct_block(model, tok, device, random.Random(args.seed), ["hobi", "umur"], "A", args.n_direct // 2)
    rep["unseen_attrs_B"] = direct_block(model, tok, device, random.Random(args.seed), ["hobi", "umur"], "B", args.n_direct // 2)
    print(json.dumps(rep, indent=1), flush=True)

    # --- tulisan tangan lewat memori + BM25 ---
    hw = [json.loads(l) for l in open("evalsets/memory_handwritten.jsonl", encoding="utf-8")]
    rng = random.Random(args.seed)
    hand_notes = list(dict.fromkeys(r["note"] for r in hw))
    synth = [G.make_fact(rng, rng.choice(G.TRAIN_ATTRS))["note"] for _ in range(300)]
    con = build_memory(hand_notes + synth)
    acc = {"oracle": 0, "top1": 0, "top3": 0}
    recall3 = 0
    rows = []
    for r in hw:
        hits = [h["text"] for h in R.retrieve(con, r["question"], 3)]
        recall3 += int(r["note"] in hits)
        outs = {"oracle": ask(model, tok, r["note"], r["question"], device),
                "top1": ask(model, tok, hits[0], r["question"], device) if hits else "",
                "top3": ask(model, tok, " ".join(hits), r["question"], device) if hits else ""}
        for k, o in outs.items():
            acc[k] += int(correct(r["answer"], o))
        rows.append({"q": r["question"], "gold": r["answer"], "hit_top3": r["note"] in hits, **outs})
    un_ref = 0
    for q in UNANSWERABLE_HW:
        hits = [h["text"] for h in R.retrieve(con, q, 3)]
        un_ref += int(refusal(ask(model, tok, " ".join(hits), q, device))) if hits else 1
    n = len(hw)
    rep["handwritten"] = {"n": n, "recall@3": recall3 / n, **{f"acc_{k}": v / n for k, v in acc.items()},
                          "refusal_unanswerable_top3": un_ref / len(UNANSWERABLE_HW), "unanswerable_n": len(UNANSWERABLE_HW)}
    print(json.dumps(rep["handwritten"], indent=1), flush=True)
    rep["handwritten_rows"] = rows
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
