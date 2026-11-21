"""
Aplikasi memori yang bisa diajari (baris perintah).

Perintah:
  ingat: <kalimat>        simpan catatan baru
  <pertanyaan>?           tanya (baris berakhiran '?', diawali kata tanya seperti 'siapa/apa/di mana', atau 'tanya: ...')
  <kalimat pernyataan>    akan ditanyakan apakah mau disimpan sebagai catatan (y/n)
  lupakan <nomor>         hapus catatan
  daftar                  tampilkan semua catatan
  impor <berkas>          impor dokumen .txt/.md (dipecah per paragraf; AI menampilkan bagian yang relevan)
  dokumen                 tampilkan dokumen yang sudah diimpor
  hapus dokumen <nomor>   hapus dokumen
  keluar                  selesai
Alat (dikerjakan program, bukan dari catatan): 'urutkan 5, 4, 3, 9', 'hitung 12 x 5', 'hari apa 17 Agustus 2026'.
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

import memory_docs
import memory_tools
from memory_agent import MemoryAgent, MemoryStore, Reader, match_rule, parse_rule

HELP = __doc__.split("Perintah:")[1].split("Contoh non-interaktif")[0]

INTRO = memory_tools.INTRO
GREETING = memory_tools.GREETING
EMPTY_MSG = ("Memori masih kosong, jadi saya belum bisa menjawab apa pun.\n"
             "Ajari saya dulu, misalnya:  ingat: Bu Rina tinggal di Jalan Mawar nomor 5.")
QUESTION_WORDS = {"apa", "apakah", "siapa", "siapakah", "kapan", "kapankah", "dimana", "dimanakah", "mana", "manakah",
                  "bagaimana", "bagaimanakah", "berapa", "berapakah", "mengapa", "kenapa", "bilamana", "kemana", "darimana"}
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
    return memory_tools.greeting_reply(len(agent.store.list()), len(agent.store.list_documents()))


def teach_message(agent, text):
    """Simpan catatan, lalu beri tahu bila dikenali sebagai aturan atau berupa tabel/rumus."""
    nid = agent.store.teach(text)
    msg = f"Dicatat [{nid}]."
    rule = parse_rule(text)
    if rule:
        msg += f' Dikenali sebagai aturan: bila Anda mengetik persis "{rule[0]}", saya menjawab "{rule[1]}".'
    if memory_tools.looks_like_table(text):
        msg += "\n  " + memory_tools.TABLE_HINT
    return msg


def format_tool(t):
    if t["kind"] == "task":
        return t["answer"]
    return t["answer"] + "\n  dihitung oleh alat \"" + t["tool"] + "\" (bukan dari catatan). " + t["detail"]


def format_passages(out, exclude_id=None):
    ps = [p for p in out.get("passages") or [] if p["id"] != exclude_id]
    if not ps:
        return []
    lines = ["  Bagian dokumen yang paling mirip:"]
    for p in ps:
        text = p["text"] if len(p["text"]) <= 240 else p["text"][:237].rstrip() + "..."
        lines.append(f"    - {p['doc']}, bagian {p['part']}: {text}")
    return lines


def import_message(agent, path):
    path = path.strip().strip('"').strip("'")
    try:
        text = memory_docs.read_text_file(path)
        name = memory_docs.sanitize_name(path)
        chunks = memory_docs.chunk_text(text)
    except memory_docs.DocumentError as e:
        return f"Gagal mengimpor: {e}"
    doc_id, parts, replaced = agent.store.add_document(name, chunks)
    return (f"Dokumen {'diganti' if replaced else 'diimpor'} [{doc_id}]: {name} ({parts} bagian).\n"
            "  Catatan: AI tidak memahami isi dokumen. Ia mencari bagian yang relevan dan menampilkannya beserta sumbernya.")


def format_answer(out):
    rel_lines = []
    if out.get("related"):
        head = ("  perhatian: ada catatan LEBIH BARU yang mirip tetapi tidak bisa saya baca; periksa sendiri:"
                if out["source"] else "  catatan terkait yang tidak bisa saya jadikan jawaban:")
        rel_lines = [head] + [f"    [{r['id']}] {r['text']}" for r in out["related"]]
    if out["status"] == "unknown":
        passages = format_passages(out)
        hint = ("  (Saya tidak bisa menyimpulkan jawabannya dengan yakin; bacalah bagian dokumen di atas sendiri.)" if passages
                else "  (Jika Anda tahu jawabannya, ajari saya dengan 'ingat: ...')")
        return "\n".join([out["answer"]] + passages + rel_lines + [hint])
    src = out["source"]
    if src.get("doc"):  # jawaban dari dokumen: bagian dokumen jadi hasil utama, jawaban AI hanya dugaan (sering keliru)
        shown = dict(out)
        shown["passages"] = [{"id": src["id"], "doc": src["doc"], "part": src["part"], "text": src["text"]}] + \
                            [p for p in out.get("passages") or [] if p["id"] != src["id"]]
        lines = format_passages(shown) + [f"  Dugaan jawaban AI (sering keliru pada dokumen; periksa bagian di atas): {out['answer']}"]
        if out["status"] == "conflict":
            lines.append("  catatan: ada catatan lain dengan jawaban berbeda:")
            lines += [f"    [{c['id']}] {c['answer']}  ({c['note']})" for c in out["conflicts"]]
        return "\n".join(lines + rel_lines)
    lines = [f"{out['answer']}", f"  sumber [{src['id']}]: {src['text']}"]
    if out["status"] == "conflict":
        lines.append("  catatan: ada catatan lain dengan jawaban berbeda; dipakai yang terbaru:")
        lines += [f"    [{c['id']}] {c['answer']}  ({c['note']})" for c in out["conflicts"]]
    return "\n".join(lines + format_passages(out, src["id"]) + rel_lines)


def handle(line, agent, confirm=None):
    """Proses satu baris masukan. confirm(prompt) -> str dipakai untuk bertanya y/n (None = mode non-interaktif)."""
    s = line.strip()
    if not s:
        return None
    low = s.lower()
    if low in ("keluar", "exit", "quit"):
        return "__exit__"
    if low in ("daftar", "/daftar"):
        rows = agent.store.list()
        return "\n".join(f"  [{i}] {t}" for i, t in rows) if rows else "  (memori masih kosong)"
    m = re.match(r"^/?impor\s+(.+)$", s, re.I)
    if m:
        return import_message(agent, m.group(1))
    if low in ("dokumen", "/dokumen"):
        docs = agent.store.list_documents()
        return "\n".join(f"  [{d['id']}] {d['name']} ({d['parts']} bagian)" for d in docs) if docs else "  (belum ada dokumen; gunakan: impor <berkas>)"
    m = re.match(r"^/?hapus\s+dokumen\s+#?(\d+)$", low)
    if m:
        return f"Dokumen [{m.group(1)}] dihapus." if agent.store.delete_document(int(m.group(1))) else f"Tidak ada dokumen aktif dengan nomor {m.group(1)}."
    m = re.match(r"^/?lupakan\s+#?(\d+)$", low)
    if m:
        return f"Catatan [{m.group(1)}] dilupakan." if agent.store.forget(int(m.group(1))) else f"Tidak ada catatan aktif dengan nomor {m.group(1)}."
    m = re.match(r"^/?ingat\s*:\s*(.+)$", s, re.I | re.S)
    if m:
        return teach_message(agent, m.group(1))
    rule = match_rule(agent.store, s)  # aturan yang Anda ajarkan didahulukan (mengalahkan sapaan bawaan)
    if rule:
        return f"{rule['reply']}\n  (aturan dari catatan [{rule['id']}])"
    if GREETING.match(s):
        return greeting_reply(agent)
    m = re.match(r"^/?tanya\s*:\s*(.+)$", s, re.I | re.S)
    tool = memory_tools.route(m.group(1) if m else s)  # alat dan permintaan tugas diperiksa sebelum pertanyaan/pernyataan
    if tool:
        return format_tool(tool)
    q = m.group(1) if m else (s if (s.endswith("?") or is_question_like(s)) else None)
    if q is not None:
        if agent.store.is_empty():
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
        return teach_message(agent, s)
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
    for stream in (sys.stdout, sys.stderr):  # konsol cp1252 tidak boleh membuat CLI macet karena karakter khusus
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    print("[memori] memuat model ...", file=sys.stderr, flush=True)
    agent = MemoryAgent(MemoryStore(args.db), Reader(args.checkpoint), k=args.k, relevance_threshold=args.relevance)
    n, nd = len(agent.store.list()), len(agent.store.list_documents())
    print(f"[memori] siap. Catatan tersimpan: {n}, dokumen: {nd}. Ketik 'bantuan' untuk perintah.", flush=True)
    print(INTRO if agent.store.is_empty() else "AI ini hanya tahu apa yang Anda ajarkan (bukan pengetahuan umum).", flush=True)

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
