"""
Ukur pemilih ALAT berbasis aturan (kode) pada dua set berlabel, tanpa model:
  dev     : evalsets/memory_router.jsonl          (dipakai untuk melihat kegagalan dan menyetel aturan)
  heldout : evalsets/memory_router_heldout.jsonl  (kegagalannya sudah dilihat sebelum menyetel: tidak lagi bersih)
  test2   : evalsets/memory_router_test2.jsonl    (dibuat setelah penyetelan; satu-satunya ukuran yang bersih)
  python tools/memory_router_rules_eval.py [--show-errors]
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from memory_router_eval import LABELS, rule_router  # noqa: E402

NOTES = [(1, "Bu Rina tinggal di Jalan Mawar nomor 5."), (2, "Skor ujian: 78, 95, 60, 88.")]


def load(name):
    return [json.loads(l) for l in open(PROJECT_ROOT / "evalsets" / name, encoding="utf-8") if l.strip()]


def main():
    show = "--show-errors" in sys.argv
    for title, name in (("dev (dipakai menyetel)", "memory_router.jsonl"), ("heldout (kegagalannya sudah saya lihat sebelum menyetel)", "memory_router_heldout.jsonl"),
                        ("test2 (BARU: ditulis setelah aturan diperbaiki, belum pernah dipakai menyetel)", "memory_router_test2.jsonl")):
        rows = load(name)
        preds = [rule_router(r["message"], NOTES) for r in rows]
        ok = sum(p == r["label"] for p, r in zip(preds, rows))
        print(f"\n== {title}: {ok}/{len(rows)} = {100 * ok / len(rows):.1f}%")
        per = {lab: [sum(p == r["label"] for p, r in zip(preds, rows) if r["label"] == lab), sum(r["label"] == lab for r in rows)] for lab in LABELS}
        print("   per kelas: " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in per.items()))
        wrong_to_tool = sum(1 for p, r in zip(preds, rows) if r["label"] == "cari" and p != "cari")
        print(f"   pesan 'cari' yang salah dibajak alat: {wrong_to_tool}")
        if show:
            for p, r in zip(preds, rows):
                if p != r["label"]:
                    print(f"     [{r['label']:7s}->{p:7s}] {r['message']}")


if __name__ == "__main__":
    main()
