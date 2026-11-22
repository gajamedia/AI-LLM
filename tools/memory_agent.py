"""
Memori yang bisa diajari: penyimpan catatan (SQLite FTS5) + pembaca (model SFT) + pagar pengaman.

Alur tanya:
  1) cari top-K catatan (BM25)
  2) model membaca TIAP catatan terpisah dan menjawab
  3) pagar pengaman: jawaban diterima hanya jika (a) bukan penolakan, (b) berupa potongan teks yang benar-benar
     ada di catatan itu, (c) opsional: semua kata bernama/angka di pertanyaan ada di catatan
  4) bila beberapa catatan menjawab berbeda, dipilih catatan terbaru; jawaban lain dilaporkan sebagai konflik
  5) jika tak ada yang lolos -> "tidak tahu" (tidak mengarang)
"""
import re
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_router
import memory_tools
import rag_lib as R

UNKNOWN = "Saya belum tahu; belum ada catatan yang menjawabnya."


_RULE = re.compile(r'^\s*(?:jika|kalau|bila|apabila)\b.*?["“]([^"“”]{1,120})["”].*?\bjawab\w*\b[^"“”]*["“]([^"“”]{1,300})["”]\s*[.!]?\s*$',
                   re.I | re.S)


def parse_rule(text):
    """Catatan berbentuk: jika ... "pemicu" ... jawab ... "balasan"  ->  (pemicu, balasan); selain itu None."""
    m = _RULE.match(text)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def match_rule(store, text):
    """Aturan dijalankan hanya bila masukan PERSIS sama dengan pemicu (tanpa membedakan huruf besar/tanda baca).
    Bila beberapa aturan cocok, yang terbaru menang."""
    key = R.norm(text)
    if not key:
        return None
    for nid, note in reversed(store.list()):
        rule = parse_rule(note)
        if rule and R.norm(rule[0]) == key:
            return {"id": nid, "text": note, "trigger": rule[0], "reply": rule[1]}
    return None


_TRAILING_BOILERPLATE = re.compile(r"\s*\b(?:berdasarkan|menurut)\s+(?:teks|informasi)\b.*$", re.I | re.S)


def clean_answer(raw):
    """Buang kalimat basa-basi yang kadang ditempelkan model setelah jawaban ('Burung Garuda Berdasarkan teks yang diberikan.').
    Bila yang tersisa kosong, jawaban dianggap penolakan."""
    return _TRAILING_BOILERPLATE.sub("", raw).strip()


def refusal(a):
    return (not a.strip()) or bool(R.REFUSE.search(a))


MAX_ANSWER_WORDS = 14          # jawaban fakta: kutipan pendek
MAX_DEFINITION_WORDS = 40      # pertanyaan definisi ("apa itu ...") memang butuh jawaban lebih panjang
_DEFINITION_Q = re.compile(r"^\s*(apa\s+itu|apakah\s+itu|apa\s+yang\s+dimaksud|apa\s+arti|apa\s+definisi|apa\s+pengertian|"
                           r"apa\s+bunyi|bagaimana\s+bunyi|jelaskan|sebutkan\s+pengertian)\b", re.I)


def answer_limit(question):
    return MAX_DEFINITION_WORDS if _DEFINITION_Q.match(question or "") else MAX_ANSWER_WORDS


READ_NEW_TOKENS = 30         # ruang menulis jawaban fakta
DEFINITION_NEW_TOKENS = 70   # jawaban definisi lebih panjang


def new_tokens(question):
    return DEFINITION_NEW_TOKENS if _DEFINITION_Q.match(question or "") else READ_NEW_TOKENS


def snap_to_word_boundaries(answer, note):
    """Model kadang berhenti di tengah kata ('Parangtriti' untuk 'Parangtritis') atau mulai di tengah kata. Bila jawaban (tanpa memperhatikan
    huruf besar-kecil) ada di catatan dan setiap kemunculannya terpotong di tepi kata, perpanjang ke tepi kata terdekat. Kemunculan yang sudah
    utuh didahulukan (jawaban tidak diubah), dan teks selain kata penuh dari catatan tidak pernah ditambahkan."""
    core = (answer or "").strip().rstrip(".")
    if len(core) < 5 or not note:                     # pecahan sangat pendek terlalu ambigu untuk diperbaiki
        return answer
    low, key = note.lower(), core.lower()
    spans, i = [], low.find(key)
    while i != -1:
        j = i + len(key)
        spans.append((i, j))
        i = low.find(key, i + 1)
    if not spans:
        return answer

    def whole(i, j):
        return (i == 0 or not note[i - 1].isalnum() or not key[0].isalnum()) and (j == len(note) or not note[j].isalnum() or not key[-1].isalnum())
    if any(whole(i, j) for i, j in spans):
        return answer
    i, j = spans[0]
    while i > 0 and note[i - 1].isalnum() and key[0].isalnum():
        i -= 1
    while j < len(note) and note[j].isalnum() and key[-1].isalnum():
        j += 1
    if (spans[0][0] - i) + (j - spans[0][1]) > 4:      # perpanjangan besar berarti bukan kata terpotong, jangan ditebak
        return answer
    return note[i:j]


def verify_reason(answer, note, max_words=MAX_ANSWER_WORDS):
    """None bila jawaban sah (potongan persis catatan dan tidak kepanjangan); selain itu alasan penolakan."""
    a = R.norm(answer.strip().rstrip("."))
    if not a:
        return "bukan kutipan persis dari catatan"
    if len(a.split()) > max_words:
        return f"terlalu panjang ({len(a.split())} kata; batas {max_words} kata untuk jawaban seperti ini)"
    return None if a in R.norm(note) else "bukan kutipan persis dari catatan"


def verified(answer, note, max_words=MAX_ANSWER_WORDS):
    return verify_reason(answer, note, max_words) is None


def _complete_list(answer, note):
    """Jawaban berupa SATU nama hari/bulan yang di catatan bagian dari daftar ('Senin sampai Rabu', 'Senin dan Selasa'): kembalikan daftar
    utuh dari catatan. None bila bukan nama hari/bulan atau tidak ada daftar."""
    a = answer.strip().rstrip(".").lower()
    for names in (_DAY_NAMES, _MONTH_NAMES):
        if a in names:
            alt = "|".join(names)
            m = re.search(rf"(?<![\w])(?:(?:{alt})\s+(?:dan|sampai|hingga|atau)\s+)*{a}(?:\s+(?:dan|sampai|hingga|atau)\s+(?:{alt}))*(?![\w])", note, re.I)
            if m and m.group(0).lower() != a:
                return m.group(0)
    return None


def complete_span(answer, note, max_extra=6):
    """Lengkapi kutipan yang berhenti di tengah nama: bila kata terakhir jawaban berhuruf kapital dan di catatan langsung
    disusul kata-kata berhuruf kapital (tanpa tanda baca), ikutkan kata-kata itu. 'Republik' -> 'Republik Indonesia'.
    Hasilnya tetap potongan persis dari catatan."""
    a = answer.strip()
    ext = _complete_list(a, note)
    if ext:
        return ext
    last = a.split()[-1] if a.split() else ""
    if not last[:1].isupper():
        return a
    for m in re.finditer(r"(?<![\w])" + re.escape(a) + r"((?:\s+[A-Z][\w-]*){1,%d})(?![\w])" % max_extra, note):
        return a + m.group(1)
    return a


def entity_ok(question, note):
    """Semua kata berhuruf kapital (selain kata pertama) dan angka di pertanyaan harus ada di catatan."""
    words = re.findall(r"[\w-]+", question)
    ents = [w for i, w in enumerate(words) if (i > 0 and w[:1].isupper()) or any(c.isdigit() for c in w)]
    nn = R.norm(note)
    return all(R.norm(e) in nn for e in ents)


_PREFIX = ("meng", "meny", "mem", "men", "me", "ber", "per", "pen", "pem", "pe", "di", "ter", "ke", "se")
_SUFFIX = ("kan", "an", "i", "nya", "lah", "kah")


def stem(w):
    w = w.lower()
    for p in _PREFIX:
        if w.startswith(p) and len(w) - len(p) >= 4:
            w = w[len(p):]
            break
    for sfx in _SUFFIX:
        if w.endswith(sfx) and len(w) - len(sfx) >= 4:
            w = w[:-len(sfx)]
            if sfx in ("nya", "lah", "kah"):  # klitik: 'ruangannya' -> 'ruangan' -> 'ruang'
                for sfx2 in ("kan", "an", "i"):
                    if w.endswith(sfx2) and len(w) - len(sfx2) >= 4:
                        w = w[:-len(sfx2)]
                        break
            break
    return w


# kata perintah di awal pertanyaan ('sebutkan lambang negara') bukan isi pertanyaan: tidak dihitung sebagai kata inti
_COMMAND_WORDS = {"sebutkan", "sebut", "jelaskan", "terangkan", "uraikan", "jabarkan", "tunjukkan", "carikan", "beritahu", "beri",
                  "katakan", "ceritakan", "tuliskan", "tolong", "coba"}
# kata jenis-jawaban ('hari apa ...', 'apa warna ...'): catatan sering berisi contohnya ('Jumat', 'merah'), bukan kata itu sendiri.
# Kata itu tidak dihitung sebagai kata inti HANYA bila jawaban model sendiri bertipe cocok (hari <- nama hari, dst.). Pelonggaran tanpa
# syarat ini terukur merugikan: menghilangkan jawaban benar yang bergantung pada kata itu dan melepas jawaban daftar yang terpotong.
_DAY_NAMES = ["senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]
_MONTH_NAMES = ["januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus", "september", "oktober", "november", "desember"]
_COLOR_NAMES = ["merah", "biru", "hijau", "kuning", "putih", "hitam", "abu-abu", "cokelat", "coklat", "ungu", "oranye", "jingga", "emas", "perak"]
_TIME_RX = re.compile(r"^\d{1,2}[.:]\d{2}(?:\s*(?:wib|wita|wit))?$", re.I)


def answer_matches_type(word, answer):
    """Apakah jawaban termasuk jenis yang ditanyakan oleh kata jenis-jawaban 'word' (hari, bulan, warna, jam/pukul, umur/usia)?"""
    a = (answer or "").strip().lower().rstrip(".")
    toks = re.findall(r"[a-z0-9.:-]+", a)
    if not toks:
        return False
    w = word.lower()
    if w == "hari":
        return all(t in _DAY_NAMES or t in ("dan", "sampai", "hingga", "atau") for t in toks) and any(t in _DAY_NAMES for t in toks)
    if w == "bulan":
        return any(t in _MONTH_NAMES for t in toks) and all(t in _MONTH_NAMES or t in ("dan", "sampai", "hingga") for t in toks)
    if w == "warna":
        return any(t in _COLOR_NAMES for t in toks) and all(t in _COLOR_NAMES or t in ("dan", "muda", "tua") for t in toks)
    if w in ("jam", "pukul"):
        return bool(_TIME_RX.match(a))
    if w in ("umur", "usia"):
        return bool(re.match(r"^\d+(?:\s*tahun)?$", a))
    return False


_TYPE_WORDS = {"hari", "warna", "jam", "pukul", "bulan", "umur", "usia"}


def relevance_score(question, note, answer=None):
    """Porsi akar kata inti non-entitas pertanyaan yang muncul di catatan (1.0 = semua). 1.0 bila tidak ada kata inti."""
    words = re.findall(r"[\w-]+", question)
    ents = {R.norm(w) for i, w in enumerate(words) if (i > 0 and w[:1].isupper()) or any(c.isdigit() for c in w)}
    nn = R.norm(note)
    # kata jenis-jawaban dibuang dari hitungan HANYA bila tidak ada di catatan dan jawaban bertipe cocok; bila ada di catatan, ia tetap
    # dihitung (cocok), sebab membuangnya justru menurunkan skor ('bulan' di catatan + jawaban 'Agustus': 3/4 menjadi 2/3 < 0,67)
    core = [stem(t) for t in R.content_tokens(question) if R.norm(t) not in ents and t.lower() not in _COMMAND_WORDS
            and not (answer and t.lower() in _TYPE_WORDS and stem(t) not in nn and answer_matches_type(t, answer))]
    core = [c for c in core if len(c) >= 4]
    if not core:
        return 1.0
    return sum(c in nn for c in core) / len(core)


def relevance_ok(question, note, threshold=0.67, answer=None):
    """Sebagian besar (>= threshold) akar kata inti non-entitas dari pertanyaan harus muncul di catatan."""
    return relevance_score(question, note, answer) >= threshold


class MemoryStore:
    """Catatan pengguna (doc_id NULL) dan potongan dokumen impor (doc_id terisi) disimpan di tabel yang sama
    agar pencarian dan pembacaan tidak berubah. list() hanya mengembalikan catatan pengguna."""

    def __init__(self, path=":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: server web melayani permintaan di banyak thread; akses diserialkan oleh pemanggil (lock).
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, created_at REAL, active INTEGER DEFAULT 1)")
        self.con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(text, tokenize='unicode61 remove_diacritics 2')")
        self._migrate()
        self.con.commit()

    def _migrate(self):
        """Memori lama (tanpa dokumen) dimigrasikan di tempat; catatan yang ada tidak berubah."""
        cols = {r[1] for r in self.con.execute("PRAGMA table_info(notes)")}
        if "doc_id" not in cols:
            self.con.execute("ALTER TABLE notes ADD COLUMN doc_id INTEGER")
        if "doc_part" not in cols:
            self.con.execute("ALTER TABLE notes ADD COLUMN doc_part INTEGER")
        self.con.execute("CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, created_at REAL, "
                         "parts INTEGER, active INTEGER DEFAULT 1)")

    def teach(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("Catatan kosong.")
        cur = self.con.execute("INSERT INTO notes(text, created_at) VALUES (?,?)", (text, time.time()))
        nid = cur.lastrowid
        self.con.execute("INSERT INTO notes_fts(rowid, text) VALUES (?,?)", (nid, text))
        self.con.commit()
        return nid

    def forget(self, note_id):
        row = self.con.execute("SELECT id FROM notes WHERE id=? AND active=1 AND doc_id IS NULL", (note_id,)).fetchone()
        if not row:
            return False
        self.con.execute("UPDATE notes SET active=0 WHERE id=?", (note_id,))
        self.con.execute("DELETE FROM notes_fts WHERE rowid=?", (note_id,))
        self.con.commit()
        return True

    def list(self):
        """Hanya catatan pengguna (bukan potongan dokumen)."""
        return self.con.execute("SELECT id, text FROM notes WHERE active=1 AND doc_id IS NULL ORDER BY id").fetchall()

    def is_empty(self):
        """Kosong bila tidak ada catatan pengguna maupun potongan dokumen."""
        return self.con.execute("SELECT 1 FROM notes WHERE active=1 LIMIT 1").fetchone() is None

    # ---------- dokumen ----------
    def add_document(self, name, chunks):
        """Impor dokumen. Dokumen aktif dengan nama sama DIGANTI. Kembalikan (doc_id, jumlah_bagian, diganti)."""
        replaced = False
        for (old_id,) in self.con.execute("SELECT id FROM documents WHERE name=? AND active=1", (name,)).fetchall():
            self._delete_document(old_id)
            replaced = True
        cur = self.con.execute("INSERT INTO documents(name, created_at, parts) VALUES (?,?,?)", (name, time.time(), len(chunks)))
        doc_id = cur.lastrowid
        now = time.time()
        for part, text in enumerate(chunks, 1):
            text = re.sub(r"\s+", " ", text).strip()
            c = self.con.execute("INSERT INTO notes(text, created_at, doc_id, doc_part) VALUES (?,?,?,?)", (text, now, doc_id, part))
            self.con.execute("INSERT INTO notes_fts(rowid, text) VALUES (?,?)", (c.lastrowid, text))
        self.con.commit()
        return doc_id, len(chunks), replaced

    def _delete_document(self, doc_id):
        ids = [r[0] for r in self.con.execute("SELECT id FROM notes WHERE doc_id=? AND active=1", (doc_id,))]
        for nid in ids:
            self.con.execute("DELETE FROM notes_fts WHERE rowid=?", (nid,))
        self.con.execute("UPDATE notes SET active=0 WHERE doc_id=?", (doc_id,))
        self.con.execute("UPDATE documents SET active=0 WHERE id=?", (doc_id,))

    def delete_document(self, doc_id):
        row = self.con.execute("SELECT id FROM documents WHERE id=? AND active=1", (doc_id,)).fetchone()
        if not row:
            return False
        self._delete_document(doc_id)
        self.con.commit()
        return True

    def list_documents(self):
        return [{"id": i, "name": n, "parts": p} for i, n, p in
                self.con.execute("SELECT id, name, parts FROM documents WHERE active=1 ORDER BY id").fetchall()]

    # ---------- pencarian ----------
    def search(self, question, k=3, docs_only=False):
        q = R.build_query(question)
        if not q:
            return []
        extra = "AND n.doc_id IS NOT NULL " if docs_only else ""
        rows = self.con.execute(
            "SELECT n.id, n.text, n.doc_id, n.doc_part, d.name FROM notes_fts f JOIN notes n ON n.id=f.rowid "
            "LEFT JOIN documents d ON d.id=n.doc_id "
            "WHERE notes_fts MATCH ? AND n.active=1 " + extra + "ORDER BY bm25(notes_fts) LIMIT ?", (q, k)).fetchall()
        return [{"id": i, "text": t, "doc_id": did, "part": part, "doc": name} for i, t, did, part, name in rows]


class Reader:
    def __init__(self, checkpoint, env="production_100m", device="cpu"):
        import torch
        from config import get_config
        from generate_gpt import _load_checkpoint_payload, _load_model
        self.torch = torch
        self.device = torch.device(device)
        cfg = get_config(env)
        ckpt = Path(checkpoint)
        payload = _load_checkpoint_payload(ckpt)
        self.model, self.tok = _load_model(ckpt, cfg, self.device, payload=payload)
        del payload

    def read(self, ctx, question, max_new=None):
        from generate_gpt import _set_seed, generate_text
        prompt = R.TEMPLATE_PROMPT.format(p=R.make_prompt(ctx, question))
        _set_seed(1)
        text = generate_text(self.model, self.tok, prompt, max_new or new_tokens(question), 0.0, 0, 1.0, 1.0, self.device)
        k = text.find("Jawaban:", max(0, len(prompt) - 20))
        return (text[k + len("Jawaban:"):] if k >= 0 else text[len(prompt):]).strip()


def echo_fraction(question, answer):
    """Porsi kata inti jawaban (akar kata) yang hanya mengulang kata inti pertanyaan. 1.0 = jawaban hanya menggemakan pertanyaan."""
    qs = {stem(t) for t in R.content_tokens(question)}
    toks = [stem(t) for t in R.content_tokens(answer)]
    return sum(t in qs for t in toks) / len(toks) if toks else 0.0


def _content_words(text):
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) >= 3 and w not in R.STOP]


def clarify_options(question, cands, best, doc_ids):
    """Kembalikan daftar pilihan bila pertanyaan ambigu; selain itu daftar kosong.
    Ambigu bila: (1) beberapa catatan sama-sama cocok (kelas pengguna/dokumen dan skor kecocokan sama dengan yang terbaik),
    (2) jawabannya berbeda, (3) tiap catatan punya kata pembeda berhuruf di luar jawabannya (mis. 'finance' vs 'produk'),
    dan (4) pertanyaan tidak memuat kata pembeda itu. Pembaruan fakta yang sama (hanya jawabannya yang berbeda) tidak ambigu."""
    group = [c for c in cands if (c["id"] in doc_ids) == (best["id"] in doc_ids) and c["score"] == best["score"]]
    by_answer = {}
    for c in group:
        by_answer.setdefault(R.norm(c["answer"]), c)          # satu pilihan per jawaban berbeda (catatan terbaru dulu)
    if len(by_answer) < 2:
        return []
    group = sorted(by_answer.values(), key=lambda c: c["id"])
    words = {c["id"]: set(_content_words(c["note"])) - set(_content_words(c["answer"])) for c in group}
    asked = set(_content_words(question))
    options = []
    for c in group:
        others = set().union(*(words[o["id"]] for o in group if o is not c))
        distinct = [w for w in dict.fromkeys(_content_words(c["note"])) if w in words[c["id"]] and w not in others and w.isalpha()]
        if not distinct or asked & set(distinct):
            return []
        options.append({"id": c["id"], "label": " ".join(distinct[:2]), "text": c["note"], "answer": c["answer"]})
    return options


_NUM_TOKEN = re.compile(r"\d+(?:[.,:]\d+)*")


def _num_shape(tok):
    return re.sub(r"\d", "9", tok)


def repair_number(raw, note):
    """Model kadang salah menyalin satu atau dua digit ('11.00' padahal catatan berisi '11.30'). Kembalikan angka persis dari catatan bila:
    jawaban berupa angka saja, bukan potongan catatan, dan di catatan ada TEPAT SATU angka berbentuk sama (pola titik/koma/titik dua dan
    jumlah digit sama) yang berbeda paling banyak 2 digit. Selain itu None (tidak menebak)."""
    a = raw.strip().rstrip(".")
    if not _NUM_TOKEN.fullmatch(a) or a in note:
        return None
    same = list(dict.fromkeys(t for t in _NUM_TOKEN.findall(note) if _num_shape(t) == _num_shape(a)))
    if len(same) != 1:
        return None
    return same[0] if sum(x != y for x, y in zip(a, same[0])) <= 2 else None


def decide(question, reads, entity_guard=True, relevance_guard=True, threshold=0.67, doc_ids=frozenset(), echo_threshold=None,
           unread_margin=None):
    """reads: daftar (id, note, raw_answer) berurutan peringkat. Kembalikan dict keputusan.
    echo_threshold: tolak jawaban yang sebagian besar katanya hanya mengulang pertanyaan (None = mati).
    unread_margin: tolak jawaban dari catatan A bila ada catatan B yang kecocokannya dengan pertanyaan lebih tinggi sedikitnya sebesar
                   ini tetapi jawaban B ditolak (model tidak bisa membacanya); A kemungkinan membahas hal lain (None = mati)."""
    cands, guesses = [], []   # guesses: jawaban mentah model yang ditolak penjaga (ditampilkan sebagai dugaan, bukan jawaban)
    for nid, note, raw in reads:
        raw = clean_answer(raw)
        if refusal(raw):
            continue
        reason = verify_reason(raw, note, answer_limit(question))
        repaired_from = None
        if reason == "bukan kutipan persis dari catatan":
            fixed = repair_number(raw, note)
            if fixed:
                repaired_from, raw = raw.strip().rstrip("."), fixed
                reason = verify_reason(raw, note, answer_limit(question))
        if reason is None and entity_guard and not entity_ok(question, note):
            reason = "nama atau angka di pertanyaan tidak ada di catatan itu"
        if reason is None and relevance_guard and not relevance_ok(question, note, threshold, raw):
            reason = "kata inti pertanyaan kurang cocok dengan catatan itu"
        if reason:
            guesses.append({"id": nid, "answer": raw.strip(), "reason": reason})
            continue
        if echo_threshold is not None and echo_fraction(question, raw) >= echo_threshold:
            guesses.append({"id": nid, "answer": raw.strip(), "reason": "jawaban hanya mengulang kata yang sudah ada di pertanyaan"})
            continue
        cands.append({"id": nid, "note": note, "answer": complete_span(raw.strip().rstrip("."), note), "repaired_from": repaired_from})
    if unread_margin is not None and cands:
        cand_ids = {c["id"] for c in cands}
        scores = {nid: relevance_score(question, note) for nid, note, _ in reads}
        kept = []
        for c in cands:
            better = [nid for nid, sc in scores.items() if nid not in cand_ids and sc - scores[c["id"]] >= unread_margin]
            if better:
                guesses.append({"id": c["id"], "answer": c["answer"], "reason": "catatan yang lebih cocok dengan pertanyaan (" +
                                ", ".join(f"#{b}" for b in better) + ") tidak bisa saya baca, jadi jawaban dari catatan ini tidak dipercaya"})
            else:
                kept.append(c)
        cands = kept
    if not cands:
        return {"status": "unknown", "answer": UNKNOWN, "source": None, "conflicts": [], "guesses": guesses, "repaired": None}
    # Urutan pemilihan: catatan pengguna didahulukan atas teks dokumen; lalu catatan yang memuat paling banyak kata inti pertanyaan
    # (agar catatan yang kebetulan lebih baru tetapi membahas hal lain tidak menang); bila sama, catatan terbaru menang (pembaruan fakta).
    for c in cands:
        c["score"] = round(relevance_score(question, c["note"]), 2)
    best = max(cands, key=lambda c: (c["id"] not in doc_ids, c["score"], c["id"]))
    options = clarify_options(question, cands, best, doc_ids)
    if options:  # beberapa catatan sama-sama cocok dan dibedakan oleh kata di luar jawaban: tanya balik, jangan memilih diam-diam
        return {"status": "clarify", "source": None, "conflicts": [], "guesses": [], "repaired": None, "options": options,
                "answer": "Ada beberapa catatan yang cocok dengan pertanyaan Anda. Maksud Anda yang mana?"}
    # catatan yang jauh kurang cocok dengan pertanyaan membahas hal lain: bukan konflik (tetap terlihat sebagai catatan terkait)
    conflicts = [c for c in cands if R.norm(c["answer"]) != R.norm(best["answer"])
                 and (unread_margin is None or best["score"] - c["score"] < unread_margin)]
    repaired = {"model": best["repaired_from"], "note": best["answer"]} if best.get("repaired_from") else None
    return {"status": "conflict" if conflicts else "ok", "answer": best["answer"],
            "source": {"id": best["id"], "text": best["note"]}, "conflicts": conflicts, "guesses": [], "repaired": repaired}


def related_unused(question, reads, used_ids, min_score=0.5):
    """Catatan top-K yang memuat semua entitas pertanyaan dan cukup cocok dengan kata intinya (>= min_score), tetapi tidak dipakai sebagai
    sumber. Syarat kecocokan mencegah catatan yang hanya kebetulan memuat satu kata sama ('mawar') tampil sebagai catatan terkait."""
    out = []
    for nid, note, raw in reads:
        if nid in used_ids:
            continue
        if entity_ok(question, note) and relevance_score(question, note) >= min_score:
            out.append({"id": nid, "text": note})
    return out


def fit_context(tok, text, question):
    """Potong catatan/bagian panjang ke jendela kalimat yang membuat PROMPT LENGKAP muat di konteks model.
    Dihitung dari prompt sebenarnya (bukan perkiraan), karena prompt terlalu panjang membuat model tidak bisa menulis jawaban."""
    def prompt_len(ctx):
        return len(tok.encode_ids(R.TEMPLATE_PROMPT.format(p=R.make_prompt(ctx, question))))
    limit = R.MAX_CONTEXT - new_tokens(question)
    if prompt_len(text) <= limit:
        return text
    best = None
    for margin in range(R.GEN_MARGIN, 201, 20):
        w = R.fit_window(tok, text, question, margin=margin)
        if not w:
            break
        best = w
        if prompt_len(w) <= limit:
            return w
    return best or text


class MemoryAgent:
    def __init__(self, store, reader, k=3, entity_guard=True, relevance_guard=True, relevance_threshold=0.67):
        self.store, self.reader, self.k = store, reader, k
        self.entity_guard, self.relevance_guard = entity_guard, relevance_guard
        self.relevance_threshold = relevance_threshold
        self.alternate_forms = True
        self.echo_threshold = 1.0      # tolak jawaban yang hanya mengulang kata pertanyaan
        self.unread_margin = 0.2       # tolak jawaban dari catatan kurang cocok bila yang lebih cocok tidak terbaca

    def _read_hits(self, hits, question):
        tok = getattr(self.reader, "tok", None)
        reads = []
        for h in hits:
            ctx = h["text"]
            if tok is not None:  # catatan/potongan panjang: ambil jendela kalimat yang membuat prompt muat di konteks model
                ctx = fit_context(tok, ctx, question)
            reads.append((h["id"], h["text"], snap_to_word_boundaries(self.reader.read(ctx, question), h["text"])))
        return reads

    def ask(self, question, history=None):
        rule = match_rule(self.store, question)  # aturan yang diajarkan pengguna didahulukan (mengalahkan sapaan bawaan)
        if rule:
            return {"status": "rule", "answer": rule["reply"], "tool": None, "detail": "",
                    "source": {"id": rule["id"], "text": rule["text"]}, "conflicts": [], "related": [], "retrieved": []}
        if memory_tools.GREETING.match(question.strip()) or memory_router.assisted_greeting(question):
            return {"status": "greeting", "answer": memory_tools.greeting_reply(len(self.store.list()), len(self.store.list_documents())), "tool": None,
                    "detail": "", "source": None, "conflicts": [], "related": [], "retrieved": []}
        notes = self.store.list()
        t = memory_router.assisted_route(question, memory_tools.parse_lexicon(notes)[0], notes)  # kamus dan angka dari catatan pengguna; classifier hanya mengusulkan (MEMORI_ROUTER)
        if t:  # alat deterministik / permintaan tugas: tidak memanggil model
            return {"status": t["kind"], "answer": t["answer"], "tool": t["tool"], "detail": t["detail"],
                    "source": None, "conflicts": [], "related": [], "retrieved": []}
        question, rewritten = memory_tools.rewrite_followup(question, history)  # pertanyaan lanjutan -> pertanyaan lengkap
        if not rewritten:  # 'gravitasi adalah', 'pengertian gravitasi', 'jelaskan gravitasi' -> 'Apa itu gravitasi?'
            question, rewritten = memory_tools.rewrite_definition(question)
        hits = self.store.search(question, self.k)
        doc_ids = frozenset(h["id"] for h in hits if h.get("doc_id") is not None)
        forms = [(question, None)]
        if self.alternate_forms:  # model terlatih dengan beberapa bentuk; bila bentuk pertama gagal, coba yang lebih sederhana
            forms += memory_tools.alternate_forms(question)
        out = reads = None
        for form, required in forms:
            form_hits = hits if required is None else [h for h in hits if R.norm(required) in R.norm(h["text"])]
            if required is not None and not form_hits:
                continue  # tidak ada catatan yang memuat kata penjelas yang dibuang: jangan menjawab dengan pertanyaan yang dilonggarkan
            form_reads = self._read_hits(form_hits, form)
            form_out = decide(form, form_reads, self.entity_guard, self.relevance_guard, self.relevance_threshold, doc_ids,
                              echo_threshold=self.echo_threshold, unread_margin=self.unread_margin)
            if out is None:
                out, reads = form_out, form_reads        # bentuk pertama: dipakai juga bila semua bentuk gagal
            if form_out["status"] != "unknown":
                out, reads = form_out, form_reads
                if form != question:
                    question, rewritten = form, True
                break
        out["retrieved"] = [h["id"] for h in hits]
        out["standalone"] = question                       # dipakai klien sebagai riwayat berikutnya
        out["rewritten"] = question if rewritten else None
        if out["source"] and out["source"]["id"] in doc_ids:
            src = next(h for h in hits if h["id"] == out["source"]["id"])
            out["source"]["doc"], out["source"]["part"] = src["doc"], src["part"]
        used = {out["source"]["id"]} if out["source"] else set()
        rel = [r for r in related_unused(question, reads, used) if r["id"] not in doc_ids]  # bagian dokumen ditampilkan di 'passages'
        if out["source"]:
            rel = [r for r in rel if r["id"] > out["source"]["id"]]  # hanya yang LEBIH BARU dari sumber
        out["related"] = rel
        out["terms"] = R.content_tokens(question)
        out["passages"] = [{"id": h["id"], "doc": h["doc"], "part": h["part"], "text": h["text"]}
                           for h in self.store.search(question, 3, docs_only=True)]
        if out["source"] and out["source"]["id"] not in doc_ids:
            out["passages"] = []  # dijawab dari catatan pengguna: bagian dokumen yang kebetulan mirip hanya jadi gangguan
        return out
