"""
Audit korpus skala (data/scale/*.u16): keutuhan berkas dan duplikat ANTAR shard.

Keutuhan: ukuran .u16 = 2 x jumlah token di manifest = 2 x jumlah .lens; jumlah dokumen .docs.npy = jumlah .lens; batas dokumen berurutan tanpa celah.
Duplikat: hash blake2b (64 bit) dari deretan token tiap dokumen (tepat sama) dan dari 128 token pertama (awalan sama, mis. dokumen yang sama dengan
ekor berbeda). Hash tiap shard disimpan di data/scale/audit_hash_<shard>.npy supaya bisa dipakai ulang.

  python tools/audit_scale_corpus.py                  # semua shard 000_0000*.u16; menulis reports/scale_corpus_audit.json
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREFIX_TOKENS = 128


def check_integrity(base):
    """base = jalur tanpa ekor, mis. data/scale/000_00001. Mengembalikan dict hasil; 'ok' True bila semua konsisten."""
    u16, lens_f, docs_f, man_f = (Path(f"{base}{e}") for e in (".u16", ".lens", ".u16.docs.npy", ".manifest.jsonl"))
    res = {"shard": Path(base).name, "ok": True, "masalah": []}
    lens = np.fromfile(lens_f, dtype=np.int32)
    docs = np.load(docs_f, mmap_mode="r")
    last = json.loads(man_f.read_text(encoding="utf-8").splitlines()[-1])
    tokens = int(lens.sum())
    res.update({"dokumen": int(lens.size), "token": tokens})
    checks = {
        "ukuran_u16_sama_2x_token": u16.stat().st_size == 2 * tokens,
        "manifest_token_sama": last["cum_tokens"] == tokens,
        "manifest_dokumen_sama": last["cum_docs"] == lens.size,
        "docs_npy_jumlah_sama": docs.shape[0] == lens.size,
        "tak_ada_dokumen_kosong": bool((lens > 0).all()),
        "batas_dokumen_tanpa_celah": bool((docs[1:, 0] == docs[:-1, 1]).all()) and int(docs[0, 0]) == 0 and int(docs[-1, 1]) == tokens,
        "panjang_sesuai_batas": bool(((docs[:, 1] - docs[:, 0]) == lens).all()),
    }
    for k, v in checks.items():
        if not v:
            res["ok"] = False
            res["masalah"].append(k)
    return res, docs


def doc_hashes(u16_path, docs):
    """(hash_penuh, hash_awalan) per dokumen; hash_awalan = 0 bila dokumen lebih pendek dari PREFIX_TOKENS."""
    arr = np.memmap(u16_path, dtype=np.uint16, mode="r")
    full = np.empty(len(docs), dtype=np.uint64)
    pre = np.zeros(len(docs), dtype=np.uint64)
    t0 = time.time()
    for i, (a, b) in enumerate(docs):
        seg = arr[a:b]
        full[i] = int.from_bytes(hashlib.blake2b(seg.tobytes(), digest_size=8).digest(), "little")
        if b - a >= PREFIX_TOKENS:
            pre[i] = int.from_bytes(hashlib.blake2b(seg[:PREFIX_TOKENS].tobytes(), digest_size=8).digest(), "little")
        if i and i % 500_000 == 0:
            print(f"    {i:,}/{len(docs):,} dokumen ({time.time() - t0:.0f} dtk)", flush=True)
    return full, pre


def dup_stats(per_shard, key):
    """per_shard: {nama: array hash}. Hitung duplikat di dalam tiap shard dan antar shard (dokumen yang hashnya sudah muncul di shard lebih awal)."""
    seen = np.empty(0, dtype=np.uint64)
    out = {}
    for name, h in per_shard.items():
        h = h[h != 0] if key == "awalan" else h
        uniq, counts = np.unique(h, return_counts=True)
        dalam = int((counts - 1).sum())
        antar = int(np.isin(uniq, seen, assume_unique=True).sum()) if seen.size else 0
        out[name] = {"dokumen_dibandingkan": int(h.size), "duplikat_di_dalam_shard": dalam, "hash_unik_sudah_ada_di_shard_sebelumnya": antar}
        seen = np.union1d(seen, uniq)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PROJECT_ROOT / "data" / "scale"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "scale_corpus_audit.json"))
    args = ap.parse_args(argv)
    d = Path(args.dir)
    names = sorted(p.name[: -len(".u16")] for p in d.glob("000_*.u16") if not p.name.endswith(".docs.npy"))
    report = {"keutuhan": [], "duplikat_penuh": None, "duplikat_awalan": None}
    fulls, pres = {}, {}
    for n in names:
        res, docs = check_integrity(d / n)
        report["keutuhan"].append(res)
        print(f"{n}: {'OK' if res['ok'] else 'MASALAH ' + str(res['masalah'])}  {res['dokumen']:,} dokumen, {res['token']:,} token", flush=True)
        cache = d / f"audit_hash_{n}.npy"
        if cache.exists():
            both = np.load(cache)
            fulls[n], pres[n] = both[0], both[1]
        else:
            fulls[n], pres[n] = doc_hashes(d / f"{n}.u16", docs)
            np.save(cache, np.stack([fulls[n], pres[n]]))
    report["duplikat_penuh"] = dup_stats(fulls, "penuh")
    report["duplikat_awalan"] = dup_stats(pres, "awalan")
    tot_tok = sum(r["token"] for r in report["keutuhan"])
    tot_doc = sum(r["dokumen"] for r in report["keutuhan"])
    dup_full = sum(v["duplikat_di_dalam_shard"] + v["hash_unik_sudah_ada_di_shard_sebelumnya"] for v in report["duplikat_penuh"].values())
    report["ringkasan"] = {"shard": len(names), "dokumen": tot_doc, "token": tot_tok, "dokumen_duplikat_penuh_kira_kira": dup_full,
                           "persen_dokumen_duplikat_penuh": round(100 * dup_full / tot_doc, 3)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["ringkasan"], ensure_ascii=False))
    print("penuh:", json.dumps(report["duplikat_penuh"], ensure_ascii=False))
    print("awalan:", json.dumps(report["duplikat_awalan"], ensure_ascii=False))
    return 0 if all(r["ok"] for r in report["keutuhan"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
