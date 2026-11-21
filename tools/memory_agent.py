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


def refusal(a):
    return (not a.strip()) or bool(R.REFUSE.search(a))


def verified(answer, note, max_words=14):
    a = R.norm(answer.strip().rstrip("."))
    if not a or len(a.split()) > max_words:
        return False
    return a in R.norm(note)


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
            break
    return w


def relevance_ok(question, note, threshold=0.67):
    """Sebagian besar (>= threshold) akar kata inti non-entitas dari pertanyaan harus muncul di catatan."""
    words = re.findall(r"[\w-]+", question)
    ents = {R.norm(w) for i, w in enumerate(words) if (i > 0 and w[:1].isupper()) or any(c.isdigit() for c in w)}
    nn = R.norm(note)
    core = [stem(t) for t in R.content_tokens(question) if R.norm(t) not in ents]
    core = [c for c in core if len(c) >= 4]
    if not core:
        return True
    return sum(c in nn for c in core) / len(core) >= threshold


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

    def read(self, ctx, question, max_new=30):
        from generate_gpt import _set_seed, generate_text
        prompt = R.TEMPLATE_PROMPT.format(p=R.make_prompt(ctx, question))
        _set_seed(1)
        text = generate_text(self.model, self.tok, prompt, max_new, 0.0, 0, 1.0, 1.0, self.device)
        k = text.find("Jawaban:", max(0, len(prompt) - 20))
        return (text[k + len("Jawaban:"):] if k >= 0 else text[len(prompt):]).strip()


def decide(question, reads, entity_guard=True, relevance_guard=True, threshold=0.67, doc_ids=frozenset()):
    """reads: daftar (id, note, raw_answer) berurutan peringkat. Kembalikan dict keputusan."""
    cands = []
    for nid, note, raw in reads:
        if refusal(raw) or not verified(raw, note):
            continue
        if entity_guard and not entity_ok(question, note):
            continue
        if relevance_guard and not relevance_ok(question, note, threshold):
            continue
        cands.append({"id": nid, "note": note, "answer": raw.strip().rstrip(".")})
    if not cands:
        return {"status": "unknown", "answer": UNKNOWN, "source": None, "conflicts": []}
    # catatan terbaru menang; catatan yang diajarkan pengguna didahulukan atas teks dokumen impor
    best = max(cands, key=lambda c: (c["id"] not in doc_ids, c["id"]))
    conflicts = [c for c in cands if R.norm(c["answer"]) != R.norm(best["answer"])]
    return {"status": "conflict" if conflicts else "ok", "answer": best["answer"],
            "source": {"id": best["id"], "text": best["note"]}, "conflicts": conflicts}


def related_unused(question, reads, used_ids):
    """Catatan top-K yang memuat semua entitas pertanyaan tetapi tidak dipakai sebagai sumber."""
    out = []
    for nid, note, raw in reads:
        if nid in used_ids:
            continue
        if entity_ok(question, note):
            out.append({"id": nid, "text": note})
    return out


READ_NEW_TOKENS = 30   # ruang minimal untuk jawaban model


def fit_context(tok, text, question):
    """Potong catatan/bagian panjang ke jendela kalimat yang membuat PROMPT LENGKAP muat di konteks model.
    Dihitung dari prompt sebenarnya (bukan perkiraan), karena prompt terlalu panjang membuat model tidak bisa menulis jawaban."""
    def prompt_len(ctx):
        return len(tok.encode_ids(R.TEMPLATE_PROMPT.format(p=R.make_prompt(ctx, question))))
    limit = R.MAX_CONTEXT - READ_NEW_TOKENS
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

    def ask(self, question):
        rule = match_rule(self.store, question)  # aturan yang diajarkan pengguna didahulukan (mengalahkan sapaan bawaan)
        if rule:
            return {"status": "rule", "answer": rule["reply"], "tool": None, "detail": "",
                    "source": {"id": rule["id"], "text": rule["text"]}, "conflicts": [], "related": [], "retrieved": []}
        if memory_tools.GREETING.match(question.strip()):
            return {"status": "greeting", "answer": memory_tools.greeting_reply(len(self.store.list()), len(self.store.list_documents())), "tool": None,
                    "detail": "", "source": None, "conflicts": [], "related": [], "retrieved": []}
        t = memory_tools.route(question, memory_tools.parse_lexicon(self.store.list())[0])  # kamus dari catatan pengguna
        if t:  # alat deterministik / permintaan tugas: tidak memanggil model
            return {"status": t["kind"], "answer": t["answer"], "tool": t["tool"], "detail": t["detail"],
                    "source": None, "conflicts": [], "related": [], "retrieved": []}
        hits = self.store.search(question, self.k)
        doc_ids = frozenset(h["id"] for h in hits if h.get("doc_id") is not None)
        tok = getattr(self.reader, "tok", None)
        reads = []
        for h in hits:
            ctx = h["text"]
            if tok is not None:  # catatan/potongan panjang: ambil jendela kalimat yang membuat prompt muat di konteks model
                ctx = fit_context(tok, ctx, question)
            reads.append((h["id"], h["text"], self.reader.read(ctx, question)))
        out = decide(question, reads, self.entity_guard, self.relevance_guard, self.relevance_threshold, doc_ids)
        out["retrieved"] = [h["id"] for h in hits]
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
