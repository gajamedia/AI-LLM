"""Tes impor dokumen: pemecahan, penyimpan (termasuk migrasi), agen, CLI, dan endpoint web (tanpa model)."""
import http.client
import json
import sqlite3
import threading

import pytest

import memory_agent as M
import memory_docs as D
from conftest import load_module

MD = """# Profil Koperasi Maju

Koperasi Maju didirikan pada 12 Maret 1998 di Desa Sukamaju. Anggotanya kini berjumlah 240 orang.

## Pengurus
- Ketua: Bu Ratna
- Bendahara: Pak Slamet

Rapat anggota diadakan setiap bulan Januari di balai desa. Pada rapat itu dibahas laporan keuangan tahunan, rencana kerja, dan pemilihan pengurus baru.

```python
print("kode tidak ikut")
```

Kontak: [situs](http://contoh.id) atau **telepon 021-555-0100**.
"""


class FakeReader:
    """Pembaca tiruan: mengutip jawaban hanya jika polanya ada di konteks; selain itu menolak."""

    def read(self, ctx, question):
        for key in ("240 orang", "Bu Ratna", "12 Maret 1998"):
            if key in ctx and any(w in question.lower() for w in ("anggota", "ketua", "didirikan")):
                return key
        return "Maaf, jawabannya tidak terdapat dalam teks yang diberikan."


def agent_with_doc(*notes):
    store = M.MemoryStore(":memory:")
    for n in notes:
        store.teach(n)
    store.add_document("profil.md", D.chunk_text(MD))
    return M.MemoryAgent(store, FakeReader())


# ---------- pemecahan ----------
def test_chunking_headings_lists_code_and_markup():
    chunks = D.chunk_text(MD)
    assert chunks[0].startswith("Profil Koperasi Maju: Koperasi Maju didirikan")
    joined = " ".join(chunks)
    assert "Ketua: Bu Ratna; Bendahara: Pak Slamet." in joined      # butir daftar dipisah titik koma
    assert "kode tidak ikut" not in joined                           # blok kode dibuang
    assert "situs" in joined and "http://" not in joined             # tautan jadi teks
    assert "telepon 021-555-0100" in joined and "**" not in joined   # tebal dibuang
    assert all(len(c) <= D.MAX_CHARS for c in chunks)


def test_long_paragraph_is_split_at_sentence_boundaries_within_limit():
    chunks = D.chunk_text("Kalimat pertama yang cukup panjang untuk uji. " * 30)
    assert len(chunks) >= 3 and all(len(c) <= D.MAX_CHARS for c in chunks)
    assert all(c.endswith(".") for c in chunks)
    assert [len(c) for c in D.chunk_text("a" * 1300)] == [500, 500, 300]  # tanpa spasi: dipotong keras


def test_short_paragraphs_merge_and_headings_without_body_vanish():
    assert D.chunk_text("# Judul A\n\n# Judul B\n\nIsi paragraf di bawah judul B yang cukup panjang untuk lolos.") == \
        ["Judul B: Isi paragraf di bawah judul B yang cukup panjang untuk lolos."]
    merged = D.chunk_text("Satu kalimat pendek di sini.\n\nDua kalimat pendek lagi di sini.")
    assert len(merged) == 1 and "Satu kalimat" in merged[0] and "Dua kalimat" in merged[0]


@pytest.mark.parametrize("bad", ["", "   \n\n  ", "x"])
def test_empty_documents_are_rejected(bad):
    with pytest.raises(D.DocumentError):
        D.chunk_text(bad)


def test_limits_and_unsupported_types(monkeypatch):
    with pytest.raises(D.DocumentError, match="belum didukung"):
        D.check_supported("laporan.pdf")
    D.check_supported("catatan.TXT")
    D.check_supported("panduan.md")
    monkeypatch.setattr(D, "MAX_CHUNKS", 2)
    with pytest.raises(D.DocumentError, match="bagian"):
        D.chunk_text("\n\n".join(f"Paragraf nomor {i} yang cukup panjang untuk menjadi bagian sendiri dalam dokumen uji ini, ya." for i in range(5)))
    monkeypatch.setattr(D, "MAX_TEXT_CHARS", 50)
    with pytest.raises(D.DocumentError, match="terlalu besar"):
        D.chunk_text("a" * 100)


def test_sanitize_name_and_decoding():
    assert D.sanitize_name("C:/Users/x/Dokumen Saya.txt") == "Dokumen Saya.txt"
    assert D.sanitize_name("../../etc/passwd") == "passwd"
    assert D.sanitize_name("a\\b\\c.md") == "c.md"
    assert D.sanitize_name("\x00\x1f") == "dokumen"
    assert D.decode_bytes("Koperasi \u2014 240".encode("utf-8")) == "Koperasi \u2014 240"
    assert D.decode_bytes("caf\xe9".encode("cp1252")) == "caf\xe9"


# ---------- penyimpan ----------
def test_old_database_is_migrated_in_place(tmp_path):
    path = str(tmp_path / "lama.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE notes(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, created_at REAL, active INTEGER DEFAULT 1)")
    con.execute("CREATE VIRTUAL TABLE notes_fts USING fts5(text, tokenize='unicode61 remove_diacritics 2')")
    con.execute("INSERT INTO notes(text, created_at) VALUES ('Bu Rina tinggal di Jalan Mawar nomor 5.', 1)")
    con.execute("INSERT INTO notes_fts(rowid, text) VALUES (1, 'Bu Rina tinggal di Jalan Mawar nomor 5.')")
    con.commit(); con.close()
    store = M.MemoryStore(path)
    assert store.list() == [(1, "Bu Rina tinggal di Jalan Mawar nomor 5.")] and store.list_documents() == []
    assert [h["id"] for h in store.search("Di mana Bu Rina tinggal?")] == [1]
    store.add_document("a.txt", ["Dokumen baru yang berisi kalimat uji cukup panjang."])
    assert len(store.list()) == 1  # catatan lama tetap, potongan dokumen tidak ikut daftar
    store.con.close()
    again = M.MemoryStore(path)     # membuka ulang tidak mengulang migrasi dengan galat
    assert len(again.list_documents()) == 1
    again.con.close()


def test_store_documents_notes_and_search_scopes():
    store = M.MemoryStore(":memory:")
    assert store.is_empty()
    store.teach("Pak Joko bekerja sebagai montir.")
    doc_id, parts, replaced = store.add_document("profil.md", D.chunk_text(MD))
    assert parts == len(D.chunk_text(MD)) and not replaced
    assert [t for _, t in store.list()] == ["Pak Joko bekerja sebagai montir."]
    hits = store.search("Kapan koperasi didirikan?")
    assert hits[0]["doc"] == "profil.md" and hits[0]["part"] == 1 and hits[0]["doc_id"] == doc_id
    assert all(h["doc_id"] is not None for h in store.search("koperasi anggota rapat", docs_only=True))
    assert store.forget(hits[0]["id"]) is False                       # potongan dokumen tidak bisa 'dilupakan' satu per satu
    new_id, new_parts, replaced = store.add_document("profil.md", ["Versi baru yang menggantikan versi lama sepenuhnya."])
    assert replaced and store.list_documents() == [{"id": new_id, "name": "profil.md", "parts": new_parts}]
    assert store.search("koperasi didirikan Maret") == []              # isi lama hilang dari pencarian
    assert store.delete_document(new_id) and not store.delete_document(new_id)
    assert not store.is_empty()                                        # masih ada catatan pengguna
    store.forget(1)
    assert store.is_empty()


def test_documents_only_memory_is_not_empty_and_rules_ignore_documents():
    store = M.MemoryStore(":memory:")
    store.add_document("aturan.txt", ['jika ada yang menyapa "halo" jawab saja "halo" adalah isi dokumen yang panjang sekali.'])
    assert not store.is_empty() and store.list() == []
    assert M.match_rule(store, "halo") is None                          # aturan hanya dari catatan pengguna
    agent = M.MemoryAgent(store, FakeReader())
    assert agent.ask("halo")["status"] == "greeting" and "1 dokumen" in agent.ask("halo")["answer"]


# ---------- agen ----------
def test_agent_returns_passages_and_cites_the_document():
    out = agent_with_doc().ask("Berapa jumlah anggota koperasi?")
    assert out["status"] == "ok" and out["answer"] == "240 orang"
    assert out["source"]["doc"] == "profil.md" and out["source"]["part"] >= 1
    assert out["passages"] and all(p["doc"] == "profil.md" for p in out["passages"])
    assert "anggota" in out["terms"]


def test_unknown_answer_still_shows_relevant_passages():
    out = agent_with_doc().ask("Siapa bendahara koperasi di desa itu sekarang?")
    assert out["status"] == "unknown"
    assert out["passages"], "bagian dokumen harus tetap ditampilkan agar pengguna bisa membacanya sendiri"
    assert any("Pak Slamet" in p["text"] for p in out["passages"])


def test_user_note_wins_over_document_text_when_both_answer():
    agent = agent_with_doc()
    agent.store.teach("Jumlah anggota koperasi sekarang adalah 300 orang.")

    class Both:
        def read(self, ctx, q):
            for key in ("300 orang", "240 orang"):
                if key in ctx:
                    return key
            return "Maaf, tidak ada."

    agent.reader = Both()
    out = agent.ask("Berapa jumlah anggota koperasi?")
    assert out["answer"] == "300 orang" and "doc" not in out["source"]     # catatan pengguna didahulukan
    assert out["status"] == "conflict" and out["conflicts"][0]["answer"] == "240 orang"


def test_document_chunks_do_not_appear_as_related_notes():
    out = agent_with_doc("Bu Ratna tinggal di Jalan Mawar.").ask("Siapa ketua Bu Ratna?")
    assert all(r["id"] not in {p["id"] for p in out["passages"]} for r in out["related"])


# ---------- CLI ----------
@pytest.fixture(scope="module")
def chat():
    return load_module("memory_chat_docs_under_test", "tools/memory_chat.py")


def test_cli_import_list_ask_and_delete(chat, tmp_path):
    f = tmp_path / "profil koperasi.md"
    f.write_text(MD, encoding="utf-8")
    agent = M.MemoryAgent(M.MemoryStore(":memory:"), FakeReader())
    out = chat.handle(f'impor "{f}"', agent)
    assert out.startswith("Dokumen diimpor [1]: profil koperasi.md") and "tidak memahami isi dokumen" in out
    assert "diganti" in chat.handle(f"impor {f}", agent)
    assert "profil koperasi.md" in chat.handle("dokumen", agent)
    answer = chat.handle("Berapa jumlah anggota koperasi?", agent)
    # jawaban dari dokumen: bagian dokumen jadi hasil utama, jawaban AI hanya "dugaan" yang diberi peringatan
    assert answer.lstrip().startswith("Bagian dokumen yang paling relevan") and "profil koperasi.md, bagian" in answer
    assert "Dugaan jawaban AI (sering keliru pada dokumen" in answer and answer.rstrip().endswith("240 orang")
    unknown = chat.handle("Siapa bendahara koperasi di desa itu sekarang?", agent)
    assert "Bagian dokumen yang paling relevan" in unknown and "Pak Slamet" in unknown and "bacalah bagian dokumen" in unknown
    assert not chat.handle("Siapa bendahara koperasi?", agent).startswith("Memori masih kosong")  # hanya-dokumen bukan 'kosong'
    assert "dihapus" in chat.handle(f"hapus dokumen {agent.store.list_documents()[0]['id']}", agent)
    assert "belum ada dokumen" in chat.handle("dokumen", agent)


def test_cli_import_errors_are_friendly(chat, tmp_path):
    agent = M.MemoryAgent(M.MemoryStore(":memory:"), FakeReader())
    assert "tidak ditemukan" in chat.handle(f"impor {tmp_path / 'tidak-ada.txt'}", agent)
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert "belum didukung" in chat.handle(f"impor {pdf}", agent)
    empty = tmp_path / "kosong.txt"
    empty.write_text("   \n", encoding="utf-8")
    assert "Tidak ada teks" in chat.handle(f"impor {empty}", agent)
    assert agent.store.list_documents() == []


# ---------- web ----------
class FakeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.store = M.MemoryStore(":memory:")
        self.agent = M.MemoryAgent(self.store, FakeReader())
        self.checkpoint = "tiruan.pt"

    def notes(self):
        return [{"id": i, "text": t} for i, t in self.store.list()]


@pytest.fixture()
def server():
    web = load_module("memory_web_docs_under_test", "tools/memory_web.py")
    state = FakeState()
    srv = web.ExclusiveServer(("127.0.0.1", 0), web.make_handler(state, 0))
    port = srv.server_address[1]
    srv.RequestHandlerClass = web.make_handler(state, port)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    def call(method, path, body=None, host=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        headers = {"Content-Type": "application/json", "Host": host or f"127.0.0.1:{port}"}
        c.request(method, path, json.dumps(body) if body is not None else None, headers)
        r = c.getresponse()
        data = json.loads(r.read().decode("utf-8") or "{}")
        c.close()
        return r.status, data

    yield call, state
    srv.shutdown()
    srv.server_close()


def test_web_document_endpoints(server):
    call, state = server
    code, j = call("POST", "/api/documents", {"name": "profil.md", "text": MD})
    assert code == 200 and j["name"] == "profil.md" and j["parts"] >= 3 and not j["replaced"]
    code, j2 = call("POST", "/api/documents", {"name": "profil.md", "text": MD})
    assert code == 200 and j2["replaced"] and len(j2["documents"]) == 1
    code, j = call("GET", "/api/documents")
    assert code == 200 and j["documents"][0]["name"] == "profil.md"
    code, j = call("GET", "/api/status")
    assert j["documents"] == 1 and j["notes"] == 0
    assert call("GET", "/api/notes")[1]["notes"] == []                   # potongan dokumen tidak tampil sebagai catatan
    code, j = call("POST", "/api/ask", {"question": "Berapa jumlah anggota koperasi?"})
    assert code == 200 and j["answer"] == "240 orang" and j["source"]["doc"] == "profil.md" and j["empty"] is False
    code, j = call("POST", "/api/ask", {"question": "Siapa bendahara koperasi di desa itu sekarang?"})
    assert j["status"] == "unknown" and j["passages"] and "Pak Slamet" in " ".join(p["text"] for p in j["passages"])
    code, j = call("DELETE", f"/api/documents/{state.store.list_documents()[0]['id']}")
    assert code == 200 and j["ok"] and j["documents"] == []
    assert call("DELETE", "/api/documents/999")[0] == 404


def test_web_document_validation_and_security(server):
    call, _ = server
    code, j = call("POST", "/api/documents", {"name": "x.pdf", "text": "isi yang cukup panjang untuk uji pdf."})
    assert code == 400 and "belum didukung" in j["error"]
    code, j = call("POST", "/api/documents", {"name": "kosong.txt", "text": "   "})
    assert code == 400 and "Tidak ada teks" in j["error"]
    code, j = call("POST", "/api/documents", {"name": "../../etc/passwd.txt", "text": "Isi dokumen yang cukup panjang untuk lolos validasi."})
    assert code == 200 and j["name"] == "passwd.txt"                    # jalur dibuang
    assert call("POST", "/api/documents", {"name": "a.txt", "text": "x"}, host="evil.example.com")[0] == 403
    assert call("GET", "/api/documents", host="evil.example.com")[0] == 403
    code, _ = call("POST", "/api/notes", {"text": "a" * 30000})        # batas catatan biasa tidak berubah
    assert code == 400
