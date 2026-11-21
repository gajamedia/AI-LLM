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

import rag_lib as R

UNKNOWN = "Saya belum tahu; belum ada catatan yang menjawabnya."


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
    def __init__(self, path=":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute("CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, created_at REAL, active INTEGER DEFAULT 1)")
        self.con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(text, tokenize='unicode61 remove_diacritics 2')")
        self.con.commit()

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
        row = self.con.execute("SELECT id FROM notes WHERE id=? AND active=1", (note_id,)).fetchone()
        if not row:
            return False
        self.con.execute("UPDATE notes SET active=0 WHERE id=?", (note_id,))
        self.con.execute("DELETE FROM notes_fts WHERE rowid=?", (note_id,))
        self.con.commit()
        return True

    def list(self):
        return self.con.execute("SELECT id, text FROM notes WHERE active=1 ORDER BY id").fetchall()

    def search(self, question, k=3):
        q = R.build_query(question)
        if not q:
            return []
        rows = self.con.execute(
            "SELECT n.id, n.text FROM notes_fts f JOIN notes n ON n.id=f.rowid "
            "WHERE notes_fts MATCH ? AND n.active=1 ORDER BY bm25(notes_fts) LIMIT ?", (q, k)).fetchall()
        return [{"id": i, "text": t} for i, t in rows]


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


def decide(question, reads, entity_guard=True, relevance_guard=True, threshold=0.67):
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
    best = max(cands, key=lambda c: c["id"])  # catatan terbaru menang
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


class MemoryAgent:
    def __init__(self, store, reader, k=3, entity_guard=True, relevance_guard=True, relevance_threshold=0.67):
        self.store, self.reader, self.k = store, reader, k
        self.entity_guard, self.relevance_guard = entity_guard, relevance_guard
        self.relevance_threshold = relevance_threshold

    def ask(self, question):
        hits = self.store.search(question, self.k)
        reads = [(h["id"], h["text"], self.reader.read(h["text"], question)) for h in hits]
        out = decide(question, reads, self.entity_guard, self.relevance_guard, self.relevance_threshold)
        out["retrieved"] = [h["id"] for h in hits]
        used = {out["source"]["id"]} if out["source"] else set()
        rel = related_unused(question, reads, used)
        if out["source"]:
            rel = [r for r in rel if r["id"] > out["source"]["id"]]  # hanya yang LEBIH BARU dari sumber
        out["related"] = rel
        return out
