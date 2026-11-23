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
        rows.append({"message": text, "label": label})
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows, bad


MENU = "  ".join(f"{i}={lab}" for i, lab in enumerate(LABELS, 1))
DEFAULT_SET = "evalsets/router_pakai_nyata.jsonl"


def _load_set(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for l in open(p, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            out.append({"message": d.get("message") or d.get("text"), "label": d["label"]})      # kunci 'message' seperti set router lain; 'text' lama diterima
    return out


def _save_set(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _pick(answer):
    a = answer.strip().lower()
    if a.isdigit() and 1 <= int(a) <= len(LABELS):
        return LABELS[int(a) - 1]
    return a if a in LABELS else None


def cmd_tulis(out=DEFAULT_SET, input_fn=input, print_fn=print):
    """Ketik pesan langsung beserta kelasnya. Pilih kelas dulu, lalu ketik pesan sebanyak yang Anda mau; baris kosong = ganti kelas; 'selesai' = simpan dan keluar."""
    rows = _load_set(out)
    have = {r["message"].strip().lower() for r in rows}
    print_fn(f"Kelas: {MENU}\nKetik angka kelas, lalu ketik pesan satu per baris seperti yang Anda ketik ke aplikasi (salah ketik dan huruf kecil boleh).\nGanti kelas: ketik /angka atau /nama (mis. /6 atau /tolak), atau baris kosong. 'selesai' = simpan dan keluar. Sudah ada {len(rows)} pesan di {out}.")
    label = None
    while True:
        try:
            line = input_fn(f"[{label or 'pilih kelas'}] > ")
        except EOFError:
            break
        s = line.strip()
        if s.lower() in ("selesai", "q", "keluar"):
            break
        if not s:
            label = None
            continue
        if s.startswith("/"):                       # '/tolak' atau '/6': ganti kelas tanpa baris kosong
            new = _pick(s[1:])
            if new:
                label = new
            else:
                print_fn(f"  pilih salah satu: {MENU}")
            continue
        if label is None:
            label = _pick(s)
            if label is None:
                print_fn(f"  pilih salah satu: {MENU}")
            continue
        if s.lower() in have:
            print_fn("  (sudah ada, dilewati)")
            continue
        rows.append({"message": s, "label": label})
        have.add(s.lower())
        _save_set(out, rows)
    print_fn(f"Tersimpan: {len(rows)} pesan, sebaran {dict(Counter(r['label'] for r in rows))} -> {out}")
    return rows


def cmd_label(log, out=DEFAULT_SET, input_fn=input, print_fn=print):
    """Beri kelas pada pesan yang tercatat di log bayangan (tanpa melihat tebakan classifier). s=lewati, h=buang pesan, q=simpan dan keluar."""
    rows = _load_set(out)
    have = {r["message"].strip().lower() for r in rows}
    todo = [t for t in read_log(log) if t.lower() not in have]
    print_fn(f"{len(todo)} pesan belum dilabeli. Kelas: {MENU}   s=lewati  h=buang (mis. pribadi)  q=simpan dan keluar")
    for i, t in enumerate(todo, 1):
        while True:
            try:
                a = input_fn(f"({i}/{len(todo)}) {t}\n  kelas? ")
            except EOFError:
                a = "q"
            a = a.strip().lower()
            if a in ("q", "selesai"):
                print_fn(f"Tersimpan: {len(rows)} pesan -> {out}")
                return rows
            if a in ("s", "h"):
                break
            lab = _pick(a)
            if lab:
                rows.append({"message": t, "label": lab})
                _save_set(out, rows)
                break
            print_fn(f"  pilih: {MENU}, s, h, atau q")
    print_fn(f"Selesai. Tersimpan: {len(rows)} pesan, sebaran {dict(Counter(r['label'] for r in rows))} -> {out}")
    return rows


def cmd_review(md="data/router/ulasan_label_150.md", out="data/router/ulasan_label_150_hasil_cli.md", input_fn=input, print_fn=print):
    """Periksa label yang saya tulis: y=ya (benar), t=tidak (salah), r=ragu (bisa dua kelas), s=lewati, q=simpan dan keluar. Kemajuan tersimpan; bisa dilanjutkan."""
    items = [(int(m.group(1)), m.group(2).strip(), m.group(3).strip()) for m in re.finditer(r"^\|\s*(\d+)\s*\|(.*)\|\s*(\w+)\s*\|[^|]*\|\s*$", Path(md).read_text(encoding="utf-8"), re.M)]
    prog_path = Path(out).with_suffix(".json")
    done = json.loads(prog_path.read_text(encoding="utf-8")) if prog_path.exists() else {}
    names = {"y": "ya", "t": "tidak", "r": "ragu"}
    print_fn(f"{len(items)} pesan, {len(done)} sudah diperiksa. Jawab: y=ya (label benar)  t=tidak  r=ragu  s=lewati  q=simpan dan keluar")
    for no, text, label in items:
        if str(no) in done:
            continue
        while True:
            try:
                a = input_fn(f"[{no}/{len(items)}] {text}\n  label: {label}   benar? (y/t/r) ")
            except EOFError:
                a = "q"
            a = a.strip().lower()
            if a in names:
                done[str(no)] = names[a]
                break
            if a == "s":
                break
            if a in ("q", "selesai"):
                _write_review(items, done, out, prog_path)
                print_fn(f"Tersimpan: {len(done)}/{len(items)} diperiksa -> {out}")
                return done
            print_fn("  jawab y, t, r, s, atau q")
    _write_review(items, done, out, prog_path)
    c = Counter(done.values())
    print_fn(f"Selesai: ya {c['ya']}, tidak {c['tidak']}, ragu {c['ragu']} dari {len(items)} -> {out}")
    return done


def _write_review(items, done, out, prog_path):
    prog_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    c = Counter(done.values())
    lines = ["# Hasil ulasan label (diisi lewat CLI)", "", f"Diperiksa {len(done)} dari {len(items)}: ya {c['ya']}, tidak {c['tidak']}, ragu {c['ragu']}.", "",
             "| No | Pesan | Label | Benar? |", "|---|---|---|---|"]
    lines += [f"| {no} | {text} | {label} | {done.get(str(no), '')} |" for no, text, label in items]
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tulis", help="ketik pesan beserta kelasnya langsung (pengganti mengisi templat)")
    t.add_argument("--out", default=DEFAULT_SET)
    l = sub.add_parser("label", help="beri kelas pada pesan yang tercatat di log bayangan")
    l.add_argument("--log", default="data/memory/router_log.jsonl")
    l.add_argument("--out", default=DEFAULT_SET)
    r = sub.add_parser("review", help="periksa 150 label yang ditulis pengembang (ya/tidak/ragu)")
    r.add_argument("--md", default="data/router/ulasan_label_150.md")
    r.add_argument("--out", default="data/router/ulasan_label_150_hasil_cli.md")
    a = sub.add_parser("to-md")
    a.add_argument("--log", default="data/memory/router_log.jsonl")
    a.add_argument("--out", default="data/router/lembar_label_pakai_nyata.md")
    b = sub.add_parser("from-md")
    b.add_argument("--md", default="data/router/lembar_label_pakai_nyata.md")
    b.add_argument("--out", default="evalsets/router_pakai_nyata.jsonl")
    args = ap.parse_args(argv)
    if args.cmd == "tulis":
        cmd_tulis(args.out)
        return 0
    if args.cmd == "label":
        cmd_label(args.log, args.out)
        return 0
    if args.cmd == "review":
        cmd_review(args.md, args.out)
        return 0
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
