"""
Mengubah log bayangan pemilih alat (pesan yang benar-benar Anda ketik) menjadi lembar label, lalu menjadi set uji independen.

1) Kumpulkan (hanya di komputer Anda; tidak dikirim ke mana pun; folder data/ tidak ikut git):
     PowerShell:  $env:MEMORI_ROUTER="shadow"; $env:MEMORI_ROUTER_LOG="data/memory/router_log.jsonl"; $env:MEMORI_ROUTER_LOG_TEXT="1"
     lalu jalankan aplikasi seperti biasa (python tools/memory_web.py atau memory_chat.py).
2) Buat lembar label (kolom Label kosong; sengaja TANPA tebakan classifier agar label Anda tidak terpengaruh):
     python tools/router_log_tools.py to-md --log data/memory/router_log.jsonl --out data/router/lembar_label_pakai_nyata.md
3) Isi kolom Label dengan salah satu: urut, hitung, tanggal, cari, sapaan, tolak (lihat penjelasan di lembar). Hapus baris yang tidak ingin dipakai.
4) Jadikan set uji:
     python tools/router_log_tools.py from-md --md data/router/lembar_label_pakai_nyata.md --out evalsets/router_pakai_nyata.jsonl
     python tools/eval_router_independent.py --file evalsets/router_pakai_nyata.jsonl --out reports/router_pakai_nyata.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

LABELS = ["urut", "hitung", "tanggal", "cari", "sapaan", "tolak"]
HEADER = """# Lembar label: pesan yang benar-benar Anda ketik

Isi kolom **Label** dengan satu kata dari daftar ini (jangan lihat dulu hasil classifier):
- `urut`: meminta angka diurutkan
- `hitung`: meminta hitungan (tambah, kurang, kali, bagi, total, rata-rata, persen)
- `tanggal`: menanyakan hari dari tanggal, atau selisih antar tanggal
- `cari`: bertanya tentang isi catatan (siapa, di mana, kapan, berapa, apa), termasuk bertanya arti kata
- `sapaan`: menyapa, terima kasih, pamit
- `tolak`: meminta sesuatu yang aplikasi ini tidak bisa (menulis, menerjemahkan, merangkum, bercerita)

Hapus baris yang tidak ingin dipakai (mis. pesan pribadi). Pesan yang bukan salah satu dari enam kelas (pernyataan biasa) sebaiknya dihapus.

| No | Pesan | Label |
|---|---|---|
"""


def read_log(path):
    seen, out = set(), []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        t = (d.get("teks") or "").strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def to_md(log, out):
    texts = read_log(log)
    rows = "".join(f"| {i} | {t.replace('|', '/')} | |\n" for i, t in enumerate(texts, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(HEADER + rows, encoding="utf-8")
    return len(texts)


def from_md(md, out):
    rows, bad = [], []
    for m in re.finditer(r"^\|\s*(\d+)\s*\|(.*)\|\s*([A-Za-z]*)\s*\|\s*$", Path(md).read_text(encoding="utf-8"), re.M):
        text, label = m.group(2).strip(), m.group(3).strip().lower()
        if not label:
            continue
        if label not in LABELS:
            bad.append((m.group(1), label))
            continue
        rows.append({"text": text, "label": label})
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("to-md")
    a.add_argument("--log", default="data/memory/router_log.jsonl")
    a.add_argument("--out", default="data/router/lembar_label_pakai_nyata.md")
    b = sub.add_parser("from-md")
    b.add_argument("--md", default="data/router/lembar_label_pakai_nyata.md")
    b.add_argument("--out", default="evalsets/router_pakai_nyata.jsonl")
    args = ap.parse_args(argv)
    if args.cmd == "to-md":
        n = to_md(args.log, args.out)
        print(f"{n} pesan unik -> {args.out}")
        return 0
    rows, bad = from_md(args.md, args.out)
    print(f"{len(rows)} pesan berlabel -> {args.out}; sebaran {dict(Counter(r['label'] for r in rows))}")
    if bad:
        print("label tidak dikenal (dilewati):", bad)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
