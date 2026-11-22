"""
Latih pemilih alat (regresi logistik pada n-gram hash) dan ukur dengan protokol yang ketat.

Protokol (ditetapkan sebelum melihat hasil):
  - Hiperparameter dan ambang dipilih HANYA dari set validasi (keluarga templat B) dan jebakan latih-tidak; set uji resmi
    (evalsets/memory_router*.jsonl) dilihat SEKALI di akhir dan TIDAK dipakai menyetel apa pun.
  - Semua angka dilaporkan dengan selang kepercayaan Wilson 95%.
  python tools/train_router.py                 # latih, pilih, ukur, simpan tools/router_weights.npz + reports/router_classifier.json
"""
import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_router as MR  # noqa: E402

LABELS = MR.LABELS


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def encode(rows):
    X = [MR.features(r["message"]) for r in rows]
    y = np.array([LABELS.index(r["label"]) for r in rows])
    return X, y


def train(X, y, l2, epochs, lr=0.05, smoothing=0.05, seed=0, batch=256):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    emb = torch.nn.EmbeddingBag(MR.N_BUCKETS, len(LABELS), mode="sum")
    torch.nn.init.zeros_(emb.weight)
    bias = torch.nn.Parameter(torch.zeros(len(LABELS)))
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    cw = torch.tensor((counts.sum() / (len(LABELS) * counts)) ** 0.5, dtype=torch.float32)       # bobot kelas ~ akar kebalikan frekuensi
    opt = torch.optim.Adam(list(emb.parameters()) + [bias], lr=lr)
    n = len(X)
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            ids = order[s:s + batch]
            flat_i = torch.from_numpy(np.concatenate([X[i][0] for i in ids]))
            flat_v = torch.from_numpy(np.concatenate([X[i][1] for i in ids]))
            offsets = torch.tensor(np.cumsum([0] + [len(X[i][0]) for i in ids[:-1]]), dtype=torch.long)
            logits = emb(flat_i, offsets, per_sample_weights=flat_v) + bias
            loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y[ids]), weight=cw, label_smoothing=smoothing)
            loss = loss + l2 * (emb.weight ** 2).sum() / len(ids)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return MR.Router(emb.weight.detach().numpy().copy(), bias.detach().numpy().copy())


def predict_all(router, rows):
    out = [router.predict(r["message"]) for r in rows]
    return [p for p, _ in out], np.array([q for _, q in out])


def metrics(rows, preds):
    n = len(rows)
    ok = sum(p == r["label"] for p, r in zip(preds, rows))
    per = {}
    for lab in LABELS:
        tp = sum(1 for p, r in zip(preds, rows) if p == lab and r["label"] == lab)
        fp = sum(1 for p, r in zip(preds, rows) if p == lab and r["label"] != lab)
        fn = sum(1 for p, r in zip(preds, rows) if p != lab and r["label"] == lab)
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        per[lab] = {"n": tp + fn, "presisi": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec) if tp else 0.0}
    present = [lab for lab in LABELS if per[lab]["n"] > 0]
    cari = [(p, r) for p, r in zip(preds, rows) if r["label"] == "cari"]
    hijack = sum(1 for p, _ in cari if p != "cari")
    return {"n": n, "benar": ok, "akurasi": ok / n, "ci": wilson(ok, n), "f1_makro": float(np.mean([per[l]["f1"] for l in present])),
            "per_kelas": per, "cari_n": len(cari), "cari_dibajak": hijack, "cari_dibajak_ci": wilson(hijack, len(cari))}


def policy_from_P(P, tau, tau_g, tau_r):
    """Kebijakan hati-hati dari matriks probabilitas: kelas non-'cari' hanya bila probabilitasnya >= ambang kelas itu; selain itu 'cari'."""
    cari = LABELS.index("cari")
    need = np.array([tau, tau, tau, 0.0, tau_g, tau_r])
    top = P.argmax(axis=1)
    keep = (top == cari) | (P[np.arange(len(P)), top] >= need[top])
    return [LABELS[t] if k else "cari" for t, k in zip(top, keep)]


def apply_policy(router, rows, tau, tau_g, tau_r):
    return policy_from_P(predict_all(router, rows)[1], tau, tau_g, tau_r)


def search_thresholds(router, va, max_hijack=0.005):
    """Cari (tau, tau_sapaan, tau_tolak) dari VALIDASI SAJA: F1 makro tertinggi dengan 'cari' dibajak <= max_hijack."""
    P = predict_all(router, va)[1]
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best = None
    for tau, tg, tr in itertools.product(grid, grid, grid):
        m = metrics(va, policy_from_P(P, tau, tg, tr))
        if m["cari_dibajak"] / max(1, m["cari_n"]) <= max_hijack:
            key = (m["f1_makro"], -(tau + tg + tr))
            if best is None or key > best[0]:
                best = (key, (tau, tg, tr), m)
    return best


def show(name, m):
    lo, hi = m["ci"]
    print(f"  {name:26s} {m['benar']:5d}/{m['n']:<5d} {100*m['akurasi']:5.1f}% [{lo:4.1f}-{hi:5.1f}]  F1 makro {m['f1_makro']:.3f}  cari dibajak {m['cari_dibajak']}/{m['cari_n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(PROJECT_ROOT / "data" / "router" / "train.jsonl"))
    ap.add_argument("--out", default=str(MR.WEIGHTS_FILE))
    ap.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "router_classifier.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ev = PROJECT_ROOT / "evalsets"
    tr, va = load(args.train), load(ev / "router_validasi.jsonl")
    je, be = load(ev / "router_jebakan.jsonl"), load(ev / "router_berisik.jsonl")
    t0 = time.time()
    Xtr, ytr = encode(tr)
    print(f"fitur latih: {len(tr)} pesan dalam {time.time()-t0:.1f} dtk")
    # --- pemilihan hiperparameter: HANYA validasi (+ jebakan sebagai penalti pembajakan) ---
    grid = list(itertools.product([1e-6, 1e-5, 1e-4], [6, 12]))
    best = None
    print("pemilihan hiperparameter (hanya validasi dan jebakan):")
    for l2, ep in grid:
        router = train(Xtr, ytr, l2, ep, seed=args.seed)
        mv = metrics(va, predict_all(router, va)[0])
        mj = metrics(je, predict_all(router, je)[0])
        score = mv["f1_makro"] - 0.5 * mj["cari_dibajak"] / max(1, mj["cari_n"])
        print(f"  l2={l2:g} epoch={ep:2d}  F1 makro validasi {mv['f1_makro']:.3f}  akurasi {100*mv['akurasi']:.1f}%  jebakan dibajak {mj['cari_dibajak']}/{mj['cari_n']}  skor {score:.3f}")
        if best is None or score > best[0]:
            best = (score, l2, ep, router)
    _, l2, ep, router = best
    print(f"dipilih: l2={l2:g}, epoch={ep}")
    # --- ambang kebijakan: pencarian sistematis dari VALIDASI saja (bukan dari set uji) ---
    found = search_thresholds(router, va)
    (tau, tau_g, tau_r), mval = (found[1], found[2]) if found else ((0.9, 0.9, 0.9), None)
    print(f"ambang dipilih dari validasi: tau={tau}, sapaan>={tau_g}, tolak>={tau_r}" + (f" | validasi: F1 makro {mval['f1_makro']:.3f}, cari dibajak {mval['cari_dibajak']}/{mval['cari_n']}" if mval else " (tidak ada kombinasi yang memenuhi batas pembajakan; dipakai 0,9)"))
    router.meta = {"l2": l2, "epoch": ep, "tau": tau, "tau_sapaan": tau_g, "tau_tolak": tau_r, "n_latih": len(tr), "seed": args.seed, "buckets": MR.N_BUCKETS}
    router.save(args.out)
    # --- pengukuran akhir: set uji resmi dilihat SEKALI ---
    report = {"hiper": {"l2": l2, "epoch": ep, "tau": tau}, "ukuran_berkas_mb": round(Path(args.out).stat().st_size / 1e6, 2)}
    print("\nPENGUKURAN (argmax mentah | dengan kebijakan ambang):")
    sets = {"validasi (templat tak terlihat)": va, "jebakan (cari)": je, "berisik (varian bergangguan)": be}
    for name, f in (("uji resmi: dev", "memory_router.jsonl"), ("uji resmi: heldout", "memory_router_heldout.jsonl"), ("uji resmi: test2", "memory_router_test2.jsonl")):
        sets[name] = load(ev / f)
    for name, rows in sets.items():
        raw = metrics(rows, predict_all(router, rows)[0])
        pol = metrics(rows, apply_policy(router, rows, tau, tau_g, tau_r))
        report[name] = {"mentah": raw, "kebijakan": pol}
        show(name + " [mentah]", raw)
        show(name + " [kebijakan]", pol)
    allacc = [r for k in ("uji resmi: dev", "uji resmi: heldout", "uji resmi: test2") for r in sets[k]]
    pol_all = apply_policy(router, allacc, tau, tau_g, tau_r)
    report["uji_resmi_gabungan"] = metrics(allacc, pol_all)
    print()
    show("uji resmi GABUNGAN", report["uji_resmi_gabungan"])
    print("\nconfusion (kebijakan, uji resmi gabungan): baris=sebenarnya, kolom=prediksi")
    cm = {a: {b: 0 for b in LABELS} for a in LABELS}
    for p, r in zip(pol_all, allacc):
        cm[r["label"]][p] += 1
    print("          " + "".join(f"{l:>9s}" for l in LABELS))
    for a in LABELS:
        print(f"  {a:8s}" + "".join(f"{cm[a][b]:>9d}" for b in LABELS))
    report["confusion_uji_resmi"] = cm
    # --- pintu lulus tahap 1 (docs/RENCANA_TAHAP1_PEMILIH_ALAT.md), dihitung otomatis ---
    pol = {k: report[k]["kebijakan"] for k in report if isinstance(report[k], dict) and "kebijakan" in report[k]}
    raw = {k: report[k]["mentah"] for k in report if isinstance(report[k], dict) and "mentah" in report[k]}
    off = [k for k in pol if k.startswith("uji resmi")]
    clean_acc = report["uji_resmi_gabungan"]["akurasi"]
    gates = [
        ("G1 uji resmi: tiap set >= 90% dan 0 'cari' dibajak", all(pol[k]["akurasi"] >= 0.90 and pol[k]["cari_dibajak"] == 0 for k in off)),
        ("G2 validasi: F1 makro >= 0,95 (kebijakan)", pol["validasi (templat tak terlihat)"]["f1_makro"] >= 0.95),
        ("G3 jebakan dialihkan <= 1% dan 'cari' dibajak <= 0,5% di validasi", pol["jebakan (cari)"]["cari_dibajak"] / pol["jebakan (cari)"]["cari_n"] <= 0.01
         and pol["validasi (templat tak terlihat)"]["cari_dibajak"] / pol["validasi (templat tak terlihat)"]["cari_n"] <= 0.005),
        ("G4 berisik: turun <= 5 poin dari uji resmi bersih", clean_acc - pol["berisik (varian bergangguan)"]["akurasi"] <= 0.05),
        ("G6 ukuran berkas <= 10 MB", report["ukuran_berkas_mb"] <= 10),
    ]
    report["pintu"] = [{"pintu": n, "lulus": bool(ok)} for n, ok in gates]
    print("\nPINTU LULUS (G5 gaya independen dan latensi diukur terpisah):")
    for n, ok in gates:
        print(f"  {'LULUS' if ok else 'GAGAL'}  {n}")
    errs = [{"set": name, "pesan": r["message"], "label": r["label"], "prediksi": p, "keluarga": r.get("family")}
            for name, rows in sets.items() for r, p in zip(rows, apply_policy(router, rows, tau, tau_g, tau_r)) if p != r["label"]]
    Path(args.report).with_name("router_classifier_kesalahan.json").write_text(json.dumps(errs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  kesalahan tersimpan: {len(errs)} pesan -> reports/router_classifier_kesalahan.json")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\nbobot: {args.out}  ({report['ukuran_berkas_mb']} MB) | laporan: {args.report}")


if __name__ == "__main__":
    main()
