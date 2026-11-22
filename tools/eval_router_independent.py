"""
Ujian gaya independen untuk pemilih alat: evalsets/router_tulisan_anda.jsonl (pesan tulisan pemilik proyek; TIDAK dipakai melatih/menyetel).
Membandingkan aturan lama, classifier murni, dan alur aplikasi (aturan + classifier mengusulkan + alat memverifikasi).
  python tools/eval_router_independent.py [--file evalsets/router_tulisan_anda.jsonl] [--out reports/router_independen.json]
Pintu G5: akurasi alur aplikasi >= 85% dan pertanyaan 'cari' yang dibajak <= 1%.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_router as MR
import memory_tools as T

NOTES = [(2, "Skor ujian: 78, 95, 60, 88.")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def label_of(result):
    return MR._rule_label(result)


def rule_label(q):
    if T.GREETING.match(q.strip()):
        return "sapaan"
    return label_of(T.route(q, None, NOTES))


def app_label(q):
    if T.GREETING.match(q.strip()) or MR.assisted_greeting(q, mode="on"):
        return "sapaan"
    return label_of(MR.assisted_route(q, None, NOTES, mode="on"))


def classifier_label(q):
    r = MR._router()
    m = r.meta
    return r.decide(q, m.get("tau", 0.5), m.get("tau_sapaan"), m.get("tau_tolak"))[0]


def score(rows, fn):
    preds = [fn(r["text"]) for r in rows]
    ok = sum(p == r["label"] for p, r in zip(preds, rows))
    cari = [(p, r) for p, r in zip(preds, rows) if r["label"] == "cari"]
    hijack = sum(p != "cari" for p, _ in cari)
    per = {}
    for lab in MR.LABELS:
        sub = [(p, r) for p, r in zip(preds, rows) if r["label"] == lab]
        per[lab] = [sum(p == lab for p, _ in sub), len(sub)]
    wrong = [{"text": r["text"], "benar": r["label"], "tebakan": p} for p, r in zip(preds, rows) if p != r["label"]]
    return {"benar": ok, "n": len(rows), "akurasi": round(ok / len(rows), 3), "ci95": wilson(ok, len(rows)), "cari_dibajak": hijack, "cari_n": len(cari), "per_kelas": per, "salah": wrong}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(PROJECT_ROOT / "evalsets" / "router_tulisan_anda.jsonl"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "router_independen.json"))
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip()]
    res = {"file": Path(args.file).name, "n": len(rows), "sebaran": dict(Counter(r["label"] for r in rows)),
           "aturan lama": score(rows, rule_label), "classifier murni": score(rows, classifier_label), "alur aplikasi (MEMORI_ROUTER=on)": score(rows, app_label)}
    app = res["alur aplikasi (MEMORI_ROUTER=on)"]
    res["pintu_G5"] = {"akurasi alur aplikasi >= 0,85": app["akurasi"] >= 0.85, "cari dibajak <= 1%": app["cari_dibajak"] <= max(0, int(0.01 * app["cari_n"]))}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("aturan lama", "classifier murni", "alur aplikasi (MEMORI_ROUTER=on)"):
        s = res[k]
        print(f"{k:36s} {s['benar']}/{s['n']} = {s['akurasi']:.1%} CI95 {s['ci95']}  cari dibajak {s['cari_dibajak']}/{s['cari_n']}  per kelas {s['per_kelas']}")
    print("pintu G5:", res["pintu_G5"])
    for w in app["salah"]:
        print("  SALAH (alur aplikasi):", w)
    return res


if __name__ == "__main__":
    main()
