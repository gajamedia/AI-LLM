"""Tes kamus buatan pengguna (kata bilangan, pengali, operator) untuk alat hitung/urut."""
import http.client
import json
import threading

import pytest

import memory_agent as M
import memory_tools as T
from conftest import load_module

NOTES = [(1, 'kata "siji" berarti 1'), (2, 'kata "loro" artinya 2'), (3, 'kata "telu" sama dengan 3'),
         (4, 'pengali "ewu" berarti 1000'), (5, 'operator "campur" berarti +')]
LEX = T.parse_lexicon(NOTES)[0]


def test_parse_valid_entries():
    lex, issues = T.parse_lexicon(NOTES)
    assert issues == []
    assert lex["units"] == {"siji": 1, "loro": 2, "telu": 3}
    assert lex["multipliers"] == {"ewu": 1000} and lex["operators"] == {"campur": "+"}


@pytest.mark.parametrize("note", [
    'kata "apa" berarti 3',                 # kata tanya
    'kata "dua" berarti 3',                 # bilangan bawaan
    'kata "dua belas" berarti 12',          # spasi
    'kata "siji" berarti 11',               # bukan satu angka
    'kata "x" berarti 1',                   # terlalu pendek
    'kata "halo" berarti 1',                # sapaan
    'kata "urutkan" berarti 1',             # perintah
    'kata "mei" berarti 5',                 # bulan
    'pengali "ewu" berarti 12',             # bukan kelipatan sepuluh
    'operator "campur" berarti ?',          # simbol tak dikenal
])
def test_invalid_entries_are_rejected_with_message(note):
    lex, issues = T.parse_lexicon([(1, note)])
    assert not any(lex.values()) and len(issues) == 1
    d = T.describe_lexicon_note(note)
    assert d and not d["ok"] and "tidak dipakai" in d["message"]


def test_non_lexicon_notes_are_ignored():
    assert T.describe_lexicon_note("Bu Rina punya kata sandi rahasia.") is None
    assert T.describe_lexicon_note("kata siji berarti 1") is None      # tanpa tanda kutip: dianggap catatan biasa
    assert T.parse_lexicon([(1, "Pak Budi tinggal di Bandung.")]) == ({"units": {}, "multipliers": {}, "operators": {}}, [])


def test_later_note_overrides_and_word_has_one_meaning():
    lex, _ = T.parse_lexicon([(1, 'kata "siji" berarti 1'), (2, 'kata "siji" berarti 7'), (3, 'operator "siji" berarti *')])
    assert lex["units"] == {} and lex["operators"] == {"siji": "*"}


def test_entry_limit():
    notes = [(i, f'kata "{"abcdefghijklmnopqrstuvwxyz"[i % 26]}{"abcdefghijklmnopqrstuvwxyz"[i // 26]}x" berarti 1') for i in range(210)]
    lex, issues = T.parse_lexicon(notes)
    assert len(lex["units"]) == T.MAX_LEXICON and len(issues) == 10


# ---------- pemakaian di alat ----------
@pytest.mark.parametrize("q,expected", [
    ("berapa siji ditambah loro", "3"),
    ("urutkan telu, siji, loro", "1, 2, 3"),
    ("urutkan telu siji loro", "1, 2, 3"),
    ("dua juta tiga ratus ribu ditambah satu", "2300001"),
    ("berapa lima ewu kali dua", "10000"),
    ("berapa 3 campur 4", "7"),
    ("berapa telu campur loro campur siji", "6"),
])
def test_tools_use_user_lexicon(q, expected):
    r = T.route(q, LEX)
    assert r and r["kind"] == "tool" and r["answer"] == expected


def test_lexicon_words_do_nothing_without_lexicon_and_do_not_hijack_sentences():
    assert T.route("berapa siji ditambah loro") is None
    assert T.route("Bu Rina punya siji anak", LEX) is None
    assert T.route("Pak Budi campur adukan semen", LEX) is None


def test_builtin_numbers_unchanged():
    f = T.words_to_digits
    assert f("seribu dua ratus tiga puluh empat") == "1234"
    assert f("dua ribu lima ratus") == "2500"
    assert f("tiga ribu") == "3000"
    assert f("seratus tiga belas") == "113"
    assert f("tiga satu dua") == "3 1 2"
    assert f("dua puluh lima ditambah satu") == "25 ditambah 1"
    assert f("dua juta seribu") == "2001000"
    assert f("sejuta lima ratus ribu") == "1500000"
    assert f("tiga miliar") == "3000000000"
    assert f("dua ribu tiga juta") == "2000 3000000"      # urutan naik: dua bilangan terpisah


# ---------- agen, CLI, web ----------
class NoReader:
    def read(self, ctx, q):
        return "Maaf, tidak ada dalam teks."


def test_agent_reads_lexicon_from_notes_and_forgetting_removes_it():
    store = M.MemoryStore(":memory:")
    agent = M.MemoryAgent(store, NoReader())
    assert agent.ask("berapa siji ditambah loro")["status"] != "tool"
    nid = store.teach('kata "siji" berarti 1')
    store.teach('kata "loro" berarti 2')
    assert agent.ask("berapa siji ditambah loro")["answer"] == "3"
    store.forget(nid)
    assert agent.ask("berapa siji ditambah loro")["status"] != "tool"


def test_cli_feedback_and_use():
    chat = load_module("memory_chat_lex_under_test", "tools/memory_chat.py")
    agent = M.MemoryAgent(M.MemoryStore(":memory:"), NoReader())
    assert "Dikenali sebagai kamus" in chat.handle('ingat: kata "siji" berarti 1', agent)
    assert "tidak dipakai sebagai kamus" in chat.handle('ingat: kata "apa" berarti 1', agent)
    chat.handle('ingat: operator "campur" berarti +', agent)
    assert chat.handle("hitung siji campur 4", agent).startswith("5")


class FakeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.store = M.MemoryStore(":memory:")
        self.agent = M.MemoryAgent(self.store, NoReader())
        self.checkpoint = "tiruan.pt"

    def notes(self):
        return [{"id": i, "text": t} for i, t in self.store.list()]


def test_web_note_feedback_and_ask():
    web = load_module("memory_web_lex_under_test", "tools/memory_web.py")
    state = FakeState()
    srv = web.ExclusiveServer(("127.0.0.1", 0), web.make_handler(state, 0))
    port = srv.server_address[1]
    srv.RequestHandlerClass = web.make_handler(state, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def call(path, body):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"})
        r = c.getresponse()
        data = json.loads(r.read().decode("utf-8"))
        c.close()
        return data

    try:
        ok = call("/api/notes", {"text": 'kata "siji" berarti 1'})
        assert ok["lexicon"]["ok"] and ok["hints"] == []
        bad = call("/api/notes", {"text": 'kata "apa" berarti 1'})
        assert bad["lexicon"]["ok"] is False
        assert call("/api/notes", {"text": "Bu Rina tinggal di Jalan Mawar nomor 5."})["lexicon"] is None
        call("/api/notes", {"text": 'operator "campur" berarti +'})
        a = call("/api/ask", {"question": "berapa siji campur 4"})
        assert a["status"] == "tool" and a["answer"] == "5"
    finally:
        srv.shutdown()
        srv.server_close()


# ---------- daftar bebas "angka = kata" ----------
def test_free_form_table_note_is_read_as_lexicon():
    note = "Kosakata Jawa: 1 = siji • 2 = loro • 3 = telu • 3 = apa • 5 = dua"
    lex, _ = T.parse_lexicon([(1, note)])
    assert lex["units"] == {"siji": 1, "loro": 2, "telu": 3}          # "apa" dan "dua" (nilai bentrok) dilewati
    d = T.describe_lexicon_note(note)
    assert d["ok"] and "3 kata baru" in d["message"] and '"apa"' in d["message"]
    assert T.route("urutkan telu siji loro", lex)["answer"] == "1, 2, 3"


def test_builtin_table_note_and_plain_equations_are_not_hijacked():
    note = "Daftar Bilangan 1 Sampai 10 • 1 = satu • 2 = dua • 3 = tiga • 10 = sepuluh"
    assert "sudah dikenal" in T.describe_lexicon_note(note)["message"]
    assert T.parse_lexicon([(1, note)])[0]["units"] == {}
    assert T.describe_lexicon_note("Rumus: 1 = 1 dan x = y") is None


# ---------- kamus bawaan dari berkas data ----------
def test_base_lexicon_comes_from_data_file_not_code():
    units, mults, ops = T.load_base_lexicon()
    assert units["tujuh"] == 7 and mults["juta"] == 1_000_000 and ops["pangkat"] == "^"
    assert (units, mults, ops) == (T._UNITS, T._MULTS, T._BASE_OPERATORS)


def test_custom_base_file_changes_what_the_tools_understand(tmp_path):
    f = tmp_path / "kamus.txt"
    f.write_text('# uji\nkata "siji" berarti 1\nkata "loro" berarti 2\npengali "ewu" berarti 1000\noperator "campur" berarti +\n', encoding="utf-8")
    units, mults, ops = T.load_base_lexicon(f)
    assert units == {"siji": 1, "loro": 2} and mults == {"ewu": 1000} and ops == {"campur": "+"}


@pytest.mark.parametrize("content", ['kata "satu" berarti 11\n', 'kata satu berarti 1\n', 'kata "satu" berarti 1\n',
                                     'pengali "ribu" berarti 12\nkata "satu" berarti 1\n', 'operator "x1" berarti +\n'])
def test_broken_base_file_fails_loudly(tmp_path, content):
    f = tmp_path / "kamus.txt"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(RuntimeError):
        T.load_base_lexicon(f)


def test_missing_base_file_gives_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="kamus_bawaan.txt"):
        T.load_base_lexicon(tmp_path / "tidak-ada.txt")


def test_package_ships_base_lexicon():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "tools" / "package_release.py").read_text(encoding="utf-8")
    assert "kamus_bawaan.txt" in text


# ---------- terbilang (angka -> huruf) ----------
@pytest.mark.parametrize("q,expected", [
    ("10", "sepuluh"), ("0", "nol"), ("7", "tujuh"), ("11", "sebelas"), ("15", "lima belas"), ("20", "dua puluh"),
    ("25", "dua puluh lima"), ("100", "seratus"), ("113", "seratus tiga belas"), ("1000", "seribu"),
    ("1.234", "seribu dua ratus tiga puluh empat"), ("2500", "dua ribu lima ratus"), ("2300001", "dua juta tiga ratus ribu satu"),
    ("1000000", "satu juta"), ("2001000", "dua juta seribu"), ("10 = ?", "sepuluh"), ("terbilang 42", "empat puluh dua"),
    ("12 dibaca apa?", "dua belas"), ("5?", "lima"),
])
def test_terbilang(q, expected):
    r = T.route(q)
    assert r and r["tool"] == "terbilang" and r["answer"] == expected


def test_terbilang_roundtrips_with_words_to_digits():
    for n in [0, 9, 10, 11, 19, 21, 99, 100, 101, 110, 999, 1000, 1001, 1999, 20000, 100000, 123456, 999999, 1000000, 2001000,
              3_000_000_000, 123_456_789_012 % 10**11]:
        assert T.words_to_digits(T.spell_number(n)) == str(n), n


def test_terbilang_edge_cases_and_user_words():
    assert T.route("1000000000000") is None                 # di luar jangkauan nama pengali
    assert T.route("tahun 2024") is None and T.route("Bu Rina umur 10 tahun") is None
    lex = T.parse_lexicon([(1, 'pengali "triliun" berarti 1000000000000')])[0]
    assert T.route("1000000000000", lex)["answer"] == "satu triliun"
    assert T.route("3", T.parse_lexicon([(1, 'kata "telu" berarti 3')])[0])["answer"] == "tiga"   # nama bawaan didahulukan


# ---------- catatan panjang: jendela konteks dan peringatan ----------
class FakeTok:
    def encode_ids(self, s, add_bos=True, add_eos=True):
        return s.split()


def test_fit_context_keeps_short_text_and_trims_long_text_around_the_question():
    q = "luas wilayah indonesia berapa?"
    assert M.fit_context(FakeTok(), "Pendek saja.", q) == "Pendek saja."
    sents = [f"Kalimat pengisi nomor {i} " + "kata " * 14 + "selesai." for i in range(30)]
    sents[22] = "Indonesia memiliki luas wilayah sebesar 1.904.569 km2."
    long_text = " ".join(sents)
    ctx = M.fit_context(FakeTok(), long_text, q)
    assert "1.904.569 km2" in ctx and len(ctx) < len(long_text)
    full = len(FakeTok().encode_ids(M.R.TEMPLATE_PROMPT.format(p=M.R.make_prompt(ctx, q))))
    assert full <= M.R.MAX_CONTEXT - M.READ_NEW_TOKENS


def test_agent_passes_trimmed_context_to_reader_for_long_user_notes():
    seen = []

    class Spy:
        tok = FakeTok()

        def read(self, ctx, q):
            seen.append(ctx)
            return "Maaf, tidak ada dalam teks."

    store = M.MemoryStore(":memory:")
    sents = [f"Kalimat pengisi nomor {i} " + "kata " * 14 + "selesai." for i in range(30)]
    sents[22] = "Indonesia memiliki luas wilayah sebesar 1.904.569 km2."
    store.teach(" ".join(sents))
    M.MemoryAgent(store, Spy()).ask("luas wilayah indonesia berapa?")
    assert seen and "1.904.569 km2" in seen[0] and len(seen[0].split()) < 250


def test_vocab_warnings():
    assert any("km2" in w for w in T.vocab_warnings("Luas Indonesia 1.904.569 km² darat."))
    assert any("karakter" in w for w in T.vocab_warnings("Kalimat panjang. " * 40))
    assert T.vocab_warnings("Bu Rina tinggal di Jalan Mawar nomor 5.") == []
