"""
Aplikasi memori yang bisa diajari (baris perintah).

Perintah:
  ingat: <kalimat>        simpan catatan baru
  <pertanyaan>?           tanya (baris berakhiran '?', diawali kata tanya seperti 'siapa/apa/di mana', atau 'tanya: ...')
  <kalimat pernyataan>    akan ditanyakan apakah mau disimpan sebagai catatan (y/n)
  lupakan <nomor>         hapus catatan
  daftar                  tampilkan semua catatan
  keluar                  selesai
PENTING: AI ini hanya tahu apa yang Anda ajarkan. Ia tidak menjawab dari pengetahuan umum, jadi ajari dulu
dengan 'ingat: ...'. Jawaban selalu disertai catatan sumber; jika tidak ada catatan yang meyakinkan, sistem
menjawab "belum tahu" dan tidak mengarang.

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

INTRO = ("AI ini hanya tahu apa yang Anda ajarkan; ia tidak menjawab dari pengetahuan umum.\n"
         "Mulai dengan:  ingat: Bu Rina tinggal di Jalan Mawar nomor 5.\n"
         "Lalu tanya:    Di mana Bu Rina tinggal?")
EMPTY_MSG = ("Memori masih kosong, jadi saya belum bisa menjawab apa pun.\n"
             "Ajari saya dulu, misalnya:  ingat: Bu Rina tinggal di Jalan Mawar nomor 5.")
QUESTION_WORDS = {"apa", "apakah", "siapa", "siapakah", "kapan", "kapankah", "dimana", "dimanakah", "mana", "manakah",
                  "bagaimana", "bagaimanakah", "berapa", "berapakah", "mengapa", "kenapa", "bilamana", "kemana", "darimana"}
GREETING = re.compile(r"^(halo|hai|hi|hei|hello|selamat\s+(pagi|siang|sore|malam)|assalamu\S*|apa\s+kabar|terima\s+kasih|makasih|thanks?)\W*$", re.I)
YES = {"y", "ya", "yes", "iya", "ok", "oke", "simpan"}


def is_question_like(s):
    """Kalimat tanya tanpa tanda '?': diawali kata tanya, atau 'di mana'."""
    words = re.findall(r"[\w'-]+", s.lower())
    if not words:
        return False
    if words[0] in QUESTION_WORDS:
        return True
    return len(words) >= 2 and words[0] == "di" and words[1] == "mana"


def greeting_reply(agent):
    n = len(agent.store.list())
    base = "Halo! Saya memori pribadi Anda: saya hanya tahu apa yang Anda ajarkan, bukan pengetahuan umum.\n"
    if n == 0:
        return base + INTRO
    return base + f"Saat ini ada {n} catatan. Tanyakan dengan kalimat berakhiran '?', atau tambah catatan dengan 'ingat: ...'."


def format_answer(out):
    rel_lines = []
    if out.get("related"):
        head = ("  perhatian: ada catatan LEBIH BARU yang mirip tetapi tidak bisa saya baca; periksa sendiri:"
                if out["source"] else "  catatan terkait yang tidak bisa saya jadikan jawaban:")
        rel_lines = [head] + [f"    [{r['id']}] {r['text']}" for r in out["related"]]
    if out["status"] == "unknown":
        hint = "  (Jika Anda tahu jawabannya, ajari saya dengan 'ingat: ...')"
        return "\n".join([out["answer"]] + rel_lines + [hint])
    lines = [f"{out['answer']}", f"  sumber [{out['source']['id']}]: {out['source']['text']}"]
    if out["status"] == "conflict":
        lines.append("  catatan: ada catatan lain dengan jawaban berbeda; dipakai yang terbaru:")
        lines += [f"    [{c['id']}] {c['answer']}  ({c['note']})" for c in out["conflicts"]]
    return "\n".join(lines + rel_lines)


def handle(line, agent, confirm=None):
    """Proses satu baris masukan. confirm(prompt) -> str dipakai untuk bertanya y/n (None = mode non-interaktif)."""
    s = line.strip()
    if not s:
        return None
    low = s.lower()
    if low in ("keluar", "exit", "quit"):
        return "__exit__"
    if GREETING.match(s):
        return greeting_reply(agent)
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
    q = m.group(1) if m else (s if (s.endswith("?") or is_question_like(s)) else None)
    if q is not None:
        if not agent.store.list():
            return EMPTY_MSG
        return format_answer(agent.ask(q))
    # bukan perintah dan bukan pertanyaan
    if len(s.split()) < 4:
        return ("Kalimat terlalu pendek untuk dipahami.\n"
                "Untuk bertanya, akhiri dengan '?'. Untuk mengajari, tulis 'ingat: <kalimat lengkap>'. Ketik 'bantuan' untuk perintah.")
    if confirm is None:
        return ("Ini tampak seperti pernyataan, tetapi tidak disimpan.\n"
                "Untuk menyimpannya tulis:  ingat: " + s)
    reply = (confirm(f'Simpan sebagai catatan?  "{s}"  (y/n): ') or "").strip().lower()
    if reply in YES:
        return f"Dicatat [{agent.store.teach(s)}]."
    return "Tidak disimpan. (Untuk bertanya, akhiri dengan '?'.)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt"))
    ap.add_argument("--db", default=str(PROJECT_ROOT / "data/memory/memory.db"))
    ap.add_argument("--cmd", action="append", help="Jalankan perintah ini lalu keluar (boleh diulang).")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--relevance", type=float, default=0.67,
                    help="Ambang kecocokan kata inti pertanyaan dengan catatan (0-1). 1.0 = ketat: lebih sering menjawab 'belum tahu', hampir tidak mengarang.")
    args = ap.parse_args()

    print("[memori] memuat model ...", file=sys.stderr, flush=True)
    agent = MemoryAgent(MemoryStore(args.db), Reader(args.checkpoint), k=args.k, relevance_threshold=args.relevance)
    n = len(agent.store.list())
    print(f"[memori] siap. Catatan tersimpan: {n}. Ketik 'bantuan' untuk perintah.", flush=True)
    print(INTRO if n == 0 else "AI ini hanya tahu apa yang Anda ajarkan (bukan pengetahuan umum).", flush=True)

    def run(line, confirm):
        if line.strip().lower() in ("bantuan", "help", "/bantuan"):
            return HELP
        return handle(line, agent, confirm)

    if args.cmd:
        for line in args.cmd:
            print(f"> {line}")
            out = run(line, None)
            if out == "__exit__":
                break
            if out:
                print(out)
        return

    def ask_yes_no(prompt):
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return "n"

    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        out = run(line, ask_yes_no)
        if out == "__exit__":
            break
        if out:
            print(out)


if __name__ == "__main__":
    main()
