"""
Aplikasi memori yang bisa diajari (baris perintah).

Perintah:
  ingat: <kalimat>        simpan catatan baru
  <pertanyaan>?           tanya (baris yang diakhiri '?', atau 'tanya: ...')
  lupakan <nomor>         hapus catatan
  daftar                  tampilkan semua catatan
  keluar                  selesai
Jawaban selalu disertai catatan sumber. Jika tidak ada catatan yang meyakinkan, sistem menjawab
"belum tahu" dan tidak mengarang.

Contoh non-interaktif:  python tools/memory_chat.py --cmd "ingat: Bu Rina tinggal di Jalan Mawar nomor 5." --cmd "Di mana Bu Rina tinggal?"
"""
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from memory_agent import MemoryAgent, MemoryStore, Reader

HELP = __doc__.split("Perintah:")[1].split("Contoh non-interaktif")[0]


def handle(line, agent):
    s = line.strip()
    if not s:
        return None
    low = s.lower()
    if low in ("keluar", "exit", "quit"):
        return "__exit__"
    if low in ("daftar", "/daftar"):
        rows = agent.store.list()
        return "\n".join(f"  [{i}] {t}" for i, t in rows) if rows else "  (memori masih kosong)"
    m = re.match(r"^/?lupakan\s+#?(\d+)$", low)
    if m:
        return f"Catatan [{m.group(1)}] dilupakan." if agent.store.forget(int(m.group(1))) else f"Tidak ada catatan aktif dengan nomor {m.group(1)}."
    m = re.match(r"^/?ingat\s*:\s*(.+)$", s, re.I | re.S)
    if m:
        nid = agent.store.teach(m.group(1))
        return f"Dicatat [{nid}]."
    m = re.match(r"^/?tanya\s*:\s*(.+)$", s, re.I | re.S)
    q = m.group(1) if m else (s if s.endswith("?") else None)
    if q is None:
        return "Tidak dikenali. Gunakan 'ingat: ...' untuk mengajari, atau akhiri pertanyaan dengan '?'. Ketik 'bantuan' untuk daftar perintah."
    out = agent.ask(q)
    rel_lines = []
    if out.get("related"):
        head = ("  perhatian: ada catatan LEBIH BARU yang mirip tetapi tidak bisa saya baca; periksa sendiri:"
                if out["source"] else "  catatan terkait yang tidak bisa saya jadikan jawaban:")
        rel_lines = [head] + [f"    [{r['id']}] {r['text']}" for r in out["related"]]
    if out["status"] == "unknown":
        return "\n".join([out["answer"]] + rel_lines)
    lines = [f"{out['answer']}", f"  sumber [{out['source']['id']}]: {out['source']['text']}"]
    if out["status"] == "conflict":
        lines.append("  catatan: ada catatan lain dengan jawaban berbeda; dipakai yang terbaru:")
        lines += [f"    [{c['id']}] {c['answer']}  ({c['note']})" for c in out["conflicts"]]
    return "\n".join(lines + rel_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="reports/checkpoints_100m_sft_mem/gpt_sft_best.pt")
    ap.add_argument("--db", default="data/memory/memory.db")
    ap.add_argument("--cmd", action="append", help="Jalankan perintah ini lalu keluar (boleh diulang).")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--relevance", type=float, default=0.67,
                    help="Ambang kecocokan kata inti pertanyaan dengan catatan (0-1). 1.0 = ketat: lebih sering menjawab 'belum tahu', hampir tidak mengarang.")
    args = ap.parse_args()

    print("[memori] memuat model ...", file=sys.stderr, flush=True)
    agent = MemoryAgent(MemoryStore(args.db), Reader(args.checkpoint), k=args.k, relevance_threshold=args.relevance)
    n = len(agent.store.list())
    print(f"[memori] siap. Catatan tersimpan: {n}. Ketik 'bantuan' untuk perintah.", flush=True)

    def run(line):
        if line.strip().lower() in ("bantuan", "help", "/bantuan"):
            return HELP
        return handle(line, agent)

    if args.cmd:
        for line in args.cmd:
            print(f"> {line}")
            out = run(line)
            if out == "__exit__":
                break
            if out:
                print(out)
        return
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        out = run(line)
        if out == "__exit__":
            break
        if out:
            print(out)


if __name__ == "__main__":
    main()
