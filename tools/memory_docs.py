"""
Impor dokumen teks (.txt, .md) ke Memori: dokumen dipecah menjadi potongan paragraf yang disimpan sebagai
catatan bertanda sumber, lalu dicari dengan BM25 seperti catatan biasa.

Catatan jujur: model 100M tidak bisa memahami dokumen. Pada uji (100 pertanyaan dari paragraf Wikipedia),
pencarian menemukan paragraf yang benar di 3 teratas pada 94% soal, tetapi jawaban akhir yang benar hanya 13%.
Karena itu fitur ini menampilkan BAGIAN DOKUMEN yang relevan lengkap dengan sumbernya; jawaban model hanya bonus.
"""
import re
from pathlib import Path

MAX_CHARS = 500          # panjang maksimal satu potongan (muat dalam konteks model 256 token)
MIN_CHARS = 20           # potongan lebih pendek dibuang
MERGE_BELOW = 120        # paragraf pendek digabung dengan paragraf berikutnya
MAX_TEXT_CHARS = 3_000_000
MAX_CHUNKS = 5000
SUPPORTED = (".txt", ".md", ".markdown", ".text")


class DocumentError(ValueError):
    """Dokumen tidak bisa diimpor (jenis tidak didukung, terlalu besar, atau kosong)."""


def sanitize_name(name):
    name = Path(str(name).replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name[:120] or "dokumen"


def check_supported(name):
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise DocumentError(f"Jenis berkas '{ext or '(tanpa ekstensi)'}' belum didukung. Yang didukung: {', '.join(SUPPORTED)}. "
                            "Untuk PDF atau DOCX, salin teksnya ke berkas .txt terlebih dahulu.")


def decode_bytes(data):
    """UTF-8 (dengan atau tanpa BOM), lalu cp1252 sebagai cadangan."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_text_file(path):
    p = Path(path)
    if not p.is_file():
        raise DocumentError(f"Berkas tidak ditemukan: {p}")
    check_supported(p.name)
    if p.stat().st_size > MAX_TEXT_CHARS * 4:
        raise DocumentError("Berkas terlalu besar.")
    return decode_bytes(p.read_bytes())


def _strip_markdown(line):
    line = re.sub(r"^\s{0,3}>\s?", "", line)                    # kutipan
    line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line)        # butir daftar
    line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)       # gambar -> teks alt
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)        # tautan -> teks
    line = re.sub(r"(\*\*|__|`)", "", line)                      # tebal / kode
    return line.strip()


def _split_long(paragraph, max_chars):
    """Pecah paragraf panjang di batas kalimat menjadi jendela <= max_chars (kalimat sangat panjang dipotong di spasi)."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", paragraph) if s]
    out, cur = [], ""
    for s in sentences:
        while len(s) > max_chars:                                # satu kalimat kepanjangan
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if cur:
                out.append(cur); cur = ""
            out.append(s[:cut].strip()); s = s[cut:].strip()
        if cur and len(cur) + 1 + len(s) > max_chars:
            out.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text, max_chars=MAX_CHARS):
    """Teks -> daftar potongan. Judul (baris '# ...' pada Markdown) menjadi awalan potongan di bawahnya."""
    if len(text) > MAX_TEXT_CHARS:
        raise DocumentError(f"Dokumen terlalu besar (maksimal {MAX_TEXT_CHARS:,} karakter).")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)           # blok kode tidak ikut
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks, heading, pending = [], "", ""

    def emit(paragraph):
        for piece in _split_long(paragraph, max_chars - (len(heading) + 2 if heading else 0)):
            piece = piece.strip()
            if len(piece) >= MIN_CHARS:
                chunks.append(f"{heading}: {piece}" if heading and not piece.startswith(heading) else piece)

    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip()]
        body, items = [], []

        def flush_items():
            if items:  # butir daftar dipisah titik koma agar tidak menyatu dengan kalimat berikutnya
                body.append("; ".join(items).rstrip(".;") + ".")
                items.clear()

        for line in lines:
            m = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
            if m:
                flush_items()
                if pending or body:
                    emit(" ".join(x for x in [pending] + body if x)); pending = ""; body = []
                heading = _strip_markdown(m.group(1))[:80]
            elif re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line):
                items.append(_strip_markdown(line))
            else:
                flush_items()
                body.append(_strip_markdown(line))
        flush_items()
        paragraph = re.sub(r"\s+", " ", " ".join(x for x in body if x)).strip()
        if not paragraph:
            continue
        pending = f"{pending} {paragraph}".strip() if pending else paragraph
        if len(pending) >= MERGE_BELOW:                           # cukup panjang: keluarkan; yang pendek digabung ke berikutnya
            emit(pending); pending = ""
    if pending:
        emit(pending)
    if not chunks:
        raise DocumentError("Tidak ada teks yang bisa diimpor dari dokumen ini.")
    if len(chunks) > MAX_CHUNKS:
        raise DocumentError(f"Dokumen menghasilkan {len(chunks):,} bagian (maksimal {MAX_CHUNKS:,}). Bagi menjadi beberapa berkas.")
    return chunks
