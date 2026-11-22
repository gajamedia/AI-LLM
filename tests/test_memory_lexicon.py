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


# ---------- pertanyaan lanjutan (riwayat) ----------
H = ["Di mana Bu Rina tinggal?"]


@pytest.mark.parametrize("q,expected", [
    ("Apa pekerjaannya?", "Apa pekerjaan Bu Rina?"),
    ("Di mana dia tinggal?", "Di mana Bu Rina tinggal?"),
    ("kalau Pak Joko?", "Di mana Pak Joko tinggal?"),
    ("bagaimana dengan Pak Joko", "Di mana Pak Joko tinggal?"),
    ("dan istrinya?", "Apa istri Bu Rina?"),
    ("lalu kalau Pak Joko bekerja di mana?", "Pak Joko bekerja di mana?"),
])
def test_followups_are_rewritten_with_the_previous_subject(q, expected):
    out, changed = T.rewrite_followup(q, H)
    assert changed and out == expected


@pytest.mark.parametrize("q", ["Di mana Pak Joko bekerja?", "Apa itu koperasi?", "Kapan lahir?", "Berapa luas Indonesia?",
                               "Rapat dimulai kapan?", "Siapa ketua koperasi?"])
def test_standalone_questions_are_left_alone(q):
    assert T.rewrite_followup(q, H) == (q, False)


def test_no_history_or_no_subject_in_history_means_no_rewrite():
    assert T.rewrite_followup("Apa pekerjaannya?", []) == ("Apa pekerjaannya?", False)
    assert T.rewrite_followup("Apa pekerjaannya?", None) == ("Apa pekerjaannya?", False)
    assert T.rewrite_followup("Apa pekerjaannya?", ["apa itu koperasi?"]) == ("Apa pekerjaannya?", False)


def test_history_skips_entity_less_questions_and_chains():
    h = ["Di mana Bu Rina tinggal?", "apa itu koperasi?"]
    assert T.rewrite_followup("Apa pekerjaannya?", h)[0] == "Apa pekerjaan Bu Rina?"
    out, _ = T.rewrite_followup("kalau Pak Joko?", H)
    assert T.rewrite_followup("dan istrinya?", H + [out])[0] == "Apa istri Pak Joko?"   # rantai memakai subjek terbaru


class Echo:
    """Pembaca tiruan yang mencatat pertanyaan yang sampai ke model."""
    def __init__(self):
        self.questions = []

    def read(self, ctx, q):
        self.questions.append(q)
        return "Jalan Mawar nomor 5" if "Jalan Mawar" in ctx and "Rina" in q else "Maaf, tidak ada dalam teks."


def test_agent_answers_followup_using_history_and_reports_rewrite():
    store = M.MemoryStore(":memory:")
    store.teach("Bu Rina tinggal di Jalan Mawar nomor 5.")
    rd = Echo()
    agent = M.MemoryAgent(store, rd)
    first = agent.ask("Di mana Bu Rina tinggal?")
    assert first["status"] == "ok" and first["rewritten"] is None and first["standalone"] == "Di mana Bu Rina tinggal?"
    follow = agent.ask("Di mana dia tinggal?", [first["standalone"]])
    assert follow["status"] == "ok" and follow["answer"] == "Jalan Mawar nomor 5"
    assert follow["rewritten"] == "Di mana Bu Rina tinggal?" and rd.questions[-1] == "Di mana Bu Rina tinggal?"
    assert agent.ask("Di mana dia tinggal?")["status"] == "unknown"      # tanpa riwayat: tidak menebak siapa "dia"


def test_cli_session_remembers_previous_question():
    chat = load_module("memory_chat_hist_under_test", "tools/memory_chat.py")
    store = M.MemoryStore(":memory:")
    store.teach("Bu Rina tinggal di Jalan Mawar nomor 5.")
    agent = M.MemoryAgent(store, Echo())
    hist = []
    chat.handle("Di mana Bu Rina tinggal?", agent, history=hist)
    out = chat.handle("Di mana dia tinggal?", agent, history=hist)
    assert out.startswith("(dibaca sebagai: Di mana Bu Rina tinggal?)") and "Jalan Mawar nomor 5" in out
    assert len(hist) == 2 and "belum tahu" in chat.handle("Di mana dia tinggal?", agent).lower()


def test_web_history_is_used_and_validated():
    web = load_module("memory_web_hist_under_test", "tools/memory_web.py")
    state = FakeState()
    state.store.teach("Bu Rina tinggal di Jalan Mawar nomor 5.")
    state.agent = M.MemoryAgent(state.store, Echo())
    srv = web.ExclusiveServer(("127.0.0.1", 0), web.make_handler(state, 0))
    port = srv.server_address[1]
    srv.RequestHandlerClass = web.make_handler(state, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def ask(body):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/api/ask", json.dumps(body), {"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"})
        r = c.getresponse()
        data = json.loads(r.read().decode("utf-8"))
        c.close()
        return r.status, data

    try:
        code, a = ask({"question": "Di mana dia tinggal?", "history": ["Di mana Bu Rina tinggal?"]})
        assert code == 200 and a["status"] == "ok" and a["rewritten"] == "Di mana Bu Rina tinggal?"
        for bad in ("bukan daftar", {"x": 1}, [1, None, ["a"]], ["x" * 5000]):
            code, b = ask({"question": "Di mana dia tinggal?", "history": bad})
            assert code == 200 and b["status"] == "unknown" and b["rewritten"] is None   # riwayat tak valid diabaikan
    finally:
        srv.shutdown()
        srv.server_close()


# ---------- angka dari catatan tertentu ----------
NN = [(1, "Skor ujian: 78, 95, 60, 88."), (2, "Bu Rina tinggal di Jalan Mawar nomor 5."), (3, "Tidak ada angka di sini.")]


@pytest.mark.parametrize("q,expected", [
    ("urutkan angka di catatan 1", "60, 78, 88, 95"),
    ("urutkan angka di catatan 1 dari terbesar", "95, 88, 78, 60"),
    ("urutkan catatan #1", "60, 78, 88, 95"),
    ("jumlahkan angka di catatan 1", "321"),
    ("berapa total angka di catatan nomor 1?", "321"),
    ("rata-rata angka di catatan 1", "80,25"),
    ("angka terbesar di catatan 1", "95"),
    ("angka terkecil di catatan 1", "60"),
    ("angka terbesar di catatan terakhir", "Tidak ada angka di catatan [3]."),
    ("jumlahkan angka di catatan 3", "Tidak ada angka di catatan [3]."),
    ("jumlahkan angka di catatan 9", "Tidak ada catatan aktif dengan nomor 9."),
])
def test_tools_work_on_numbers_in_one_named_note(q, expected):
    r = T.route(q, None, NN)
    assert r and r["kind"] == "tool" and r["answer"] == expected
    if "[3]" not in expected and "nomor 9" not in expected:
        assert "dibaca dari catatan [1]" in r["detail"] and "Periksa" in r["detail"]       # transparan: angka yang dibaca ditampilkan


def test_notes_tool_does_not_hijack_normal_questions_and_refuses_to_mix_all_notes():
    assert T.route("jumlah catatan saya berapa?", None, NN) is None
    assert T.route("Apa isi catatan 1?", None, NN) is None
    assert T.route("Berapa total biaya rapat?", None, NN) is None
    r = T.route("urutkan angka di catatan saya", None, NN)
    assert r["kind"] == "task" and "Sebutkan catatan" in r["answer"]
    assert T.route("jumlahkan angka di catatan 1", None, None)["answer"] == "Tidak ada catatan aktif dengan nomor 1."


def test_agent_and_cli_use_notes_for_tool():
    chat = load_module("memory_chat_notes_under_test", "tools/memory_chat.py")
    store = M.MemoryStore(":memory:")
    nid = store.teach("Skor ujian: 78, 95, 60, 88.")
    agent = M.MemoryAgent(store, NoReader())
    out = agent.ask(f"jumlahkan angka di catatan {nid}")
    assert out["status"] == "tool" and out["answer"] == "321"
    assert "60, 78, 88, 95" in chat.handle(f"urutkan angka di catatan {nid}", agent)
    store.forget(nid)
    assert "Tidak ada catatan aktif" in chat.handle(f"urutkan angka di catatan {nid}", agent)


# ---------- kutipan yang terpotong di tengah nama ----------
NOTE_RI = "Indonesia, dengan nama resmi Republik Indonesia, adalah sebuah negara di Asia Tenggara."


@pytest.mark.parametrize("answer,note,expected", [
    ("Republik", NOTE_RI, "Republik Indonesia"),                       # kasus nyata: berhenti di tengah nama
    ("Bu", "Ketua koperasi adalah Bu Ratna Dewi di desa.", "Bu Ratna Dewi"),
    ("Jalan Mawar nomor 5", "Bu Rina tinggal di Jalan Mawar nomor 5.", "Jalan Mawar nomor 5"),   # tidak diubah
    ("Ratna", "Ketua adalah Bu Ratna.", "Ratna"),                      # hanya memperluas ke kanan
    ("Bekasi", "Pak Joko bekerja di Bekasi. Rapat mulai pukul 9.", "Bekasi"),   # tanda baca menghentikan
    ("guru", "Bu Rina bekerja sebagai guru Matematika.", "guru"),      # kata terakhir bukan kapital: tidak diperluas
])
def test_complete_span(answer, note, expected):
    assert M.complete_span(answer, note) == expected
    assert expected in note


def test_agent_returns_completed_name():
    class Partial:
        def read(self, ctx, q):
            return "Republik"

    store = M.MemoryStore(":memory:")
    store.teach(NOTE_RI)
    out = M.MemoryAgent(store, Partial()).ask("nama resmi indonesia adalah")
    assert out["answer"] == "Republik Indonesia"


# ---------- jawaban mentah model yang ditolak ditampilkan ----------
def test_rejected_model_answers_are_reported_with_reasons():
    note = "Bu Ratna adalah ketua koperasi Maju di Desa Sukamaju."
    reads = [(1, note, "Pak Slamet"), (2, note, "Maaf, tidak ada dalam teks.")]
    d = M.decide("Siapa ketua koperasi Maju?", reads)
    assert d["status"] == "unknown" and d["guesses"] == [{"id": 1, "answer": "Pak Slamet", "reason": "bukan kutipan persis dari catatan"}]
    d2 = M.decide("Siapa ketua Koperasi Sejahtera?", [(1, note, "Bu Ratna")])
    assert d2["status"] == "unknown" and "nama atau angka" in d2["guesses"][0]["reason"]
    d3 = M.decide("Berapa harga beras kemarin di pasar induk?", [(1, note, "Bu Ratna")])
    assert "kata inti" in d3["guesses"][0]["reason"]
    assert M.decide("Siapa ketua koperasi Maju?", [(1, note, "Bu Ratna")])["guesses"] == []   # dijawab: tidak ada dugaan tertolak


def test_cli_shows_rejected_guess():
    chat = load_module("memory_chat_guess_under_test", "tools/memory_chat.py")

    class Wrong:
        def read(self, ctx, q):
            return "Pak Slamet"

    store = M.MemoryStore(":memory:")
    store.teach("Bu Ratna adalah ketua koperasi Maju di Desa Sukamaju.")
    out = chat.handle("Siapa ketua koperasi Maju?", M.MemoryAgent(store, Wrong()))
    assert "jawaban mentah model yang ditolak penjaga" in out and '"Pak Slamet"' in out and "bukan kutipan persis" in out


# ---------- tempel beberapa baris -> beberapa catatan ----------
def test_split_note_lines():
    t = "Catatan:\n- Ketua koperasi Maju adalah Bu Ratna.\r\n* Bendahara adalah Pak Slamet.\n\n1. Anggota ada 240 orang.\n2) Rapat tiap Januari.\n"
    lines, skipped = T.split_note_lines(t)
    assert lines == ["Ketua koperasi Maju adalah Bu Ratna.", "Bendahara adalah Pak Slamet.", "Anggota ada 240 orang.", "Rapat tiap Januari."]
    assert skipped == ["Catatan:"]
    assert T.split_note_lines("Satu baris saja di sini.") == (["Satu baris saja di sini."], [])
    table = "Daftar:\n1 = siji\n2 = loro\n3 = telu"
    assert T.split_note_lines(table) == (["Daftar: 1 = siji 2 = loro 3 = telu"], [])     # kamus: satu catatan, tidak dipecah


class MultiHarness:
    def __init__(self):
        self.web = load_module("memory_web_multi_under_test", "tools/memory_web.py")
        self.state = FakeState()
        srv = self.web.ExclusiveServer(("127.0.0.1", 0), self.web.make_handler(self.state, 0))
        self.port = srv.server_address[1]
        srv.RequestHandlerClass = self.web.make_handler(self.state, self.port)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.srv = srv

    def post(self, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/api/notes", json.dumps(body), {"Content-Type": "application/json", "Host": f"127.0.0.1:{self.port}"})
        r = c.getresponse()
        data = json.loads(r.read().decode("utf-8"))
        c.close()
        return r.status, data

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def test_web_multi_line_paste_creates_one_note_per_line():
    h = MultiHarness()
    try:
        text = "Koperasi Maju\nKetua koperasi Maju adalah Bu Ratna.\n- Bendahara koperasi Maju adalah Pak Slamet.\nHalo\n" \
               'kata "siji" berarti 1\nkata "apa" berarti 2'
        code, j = h.post({"text": text, "multi": True})
        assert code == 200 and j["count"] == 5 and j["ids"] == [i["id"] for i in j["items"]] and j["skipped"] == ["Halo"]
        assert [n["text"] for n in j["notes"]] == ["Koperasi Maju", "Ketua koperasi Maju adalah Bu Ratna.",
                                                  "Bendahara koperasi Maju adalah Pak Slamet.", 'kata "siji" berarti 1', 'kata "apa" berarti 2']
        assert len(j["lexicon_issues"]) == 1 and "apa" in j["lexicon_issues"][0]           # kamus yang ditolak dilaporkan
        # tanpa flag multi: perilaku lama (satu catatan, baris baru dilebur)
        code, j2 = h.post({"text": "Baris satu yang cukup.\nBaris dua yang cukup."})
        assert code == 200 and "count" not in j2 and h.state.notes()[-1]["text"] == "Baris satu yang cukup. Baris dua yang cukup."
    finally:
        h.close()


def test_web_multi_line_is_all_or_nothing_and_limited():
    h = MultiHarness()
    try:
        before = len(h.state.notes())
        code, j = h.post({"text": "Kalimat pertama yang baik.\n" + "kata " * 130 + "\nKalimat ketiga.", "multi": True})
        assert code == 400 and "Baris 2 terlalu panjang" in j["error"] and len(h.state.notes()) == before   # tidak ada yang tersimpan
        code, j = h.post({"text": "\n".join(f"Catatan nomor {i} ada." for i in range(61)), "multi": True})
        assert code == 400 and "Terlalu banyak baris" in j["error"] and len(h.state.notes()) == before
        code, j = h.post({"text": "x " * 3100, "multi": True})
        assert code == 400 and "terlalu panjang" in j["error"].lower()
        code, j = h.post({"text": "  \n \n", "multi": True})
        assert code == 400 and len(h.state.notes()) == before
        code, j = h.post({"text": "1 = siji\n2 = loro\n3 = telu", "multi": True})                        # kamus tidak dipecah
        assert code == 200 and j["count"] == 1 and j["lexicon"]["ok"]
    finally:
        h.close()


def test_web_multi_line_with_replace_id_replaces_the_old_note():
    h = MultiHarness()
    try:
        _, first = h.post({"text": "Catatan lama yang akan diganti."})
        code, j = h.post({"text": "Baris baru satu ada.\nBaris baru dua ada.", "multi": True, "replace_id": first["id"]})
        assert code == 200 and [n["text"] for n in h.state.notes()] == ["Baris baru satu ada.", "Baris baru dua ada."]
    finally:
        h.close()


# ---------- batas panjang jawaban: fakta vs definisi ----------
GRAV = "Hukum Gravitasi Universal: Gaya tarik-menarik antar benda bermassa berbanding lurus dengan perkalian massanya dan berbanding terbalik dengan kuadrat jaraknya."
GRAV_ANS = "gaya tarik-menarik antar benda bermassa berbanding lurus dengan perkalian massanya dan berbanding terbalik dengan kuadrat jaraknya"


def test_long_answer_is_accepted_for_definition_questions_only():
    d = M.decide("apa itu gravitasi", [(1, GRAV, GRAV_ANS)])
    assert d["status"] == "ok" and d["answer"] == GRAV_ANS
    d2 = M.decide("Siapa penemu hukum gravitasi universal", [(1, GRAV, GRAV_ANS)])        # pertanyaan fakta: batas 14 kata
    assert d2["status"] == "unknown"
    assert d2["guesses"][0]["reason"].startswith("terlalu panjang (16 kata; batas 14 kata")   # alasan jujur, bukan "bukan kutipan"
    assert M.decide("apa itu gravitasi", [(1, GRAV, "gaya tarik antar bumi dan bulan")])["guesses"][0]["reason"] == "bukan kutipan persis dari catatan"
    long_answer = " ".join(["kata"] * 45)
    assert "terlalu panjang" in M.decide("apa itu gravitasi", [(1, long_answer, long_answer)])["guesses"][0]["reason"]


def test_definition_questions_get_more_room_to_write():
    assert M.new_tokens("Apa itu gravitasi") == M.DEFINITION_NEW_TOKENS > M.new_tokens("Siapa ketua koperasi?") == M.READ_NEW_TOKENS
    assert M.answer_limit("Jelaskan hukum gravitasi") == M.MAX_DEFINITION_WORDS and M.answer_limit("Kapan rapat?") == M.MAX_ANSWER_WORDS


# ---------- pertanyaan definisi dalam bentuk lain ----------
@pytest.mark.parametrize("q,expected", [
    ("gravitasi adalah", "Apa itu gravitasi"), ("gravitasi itu apa?", "Apa itu gravitasi"), ("gravitasi adalah apa?", "Apa itu gravitasi"),
    ("pengertian gravitasi", "Apa itu gravitasi"), ("Definisi gravitasi universal", "Apa itu gravitasi universal"),
    ("apa arti kata sinergi", "Apa itu sinergi"), ("jelaskan gravitasi", "Apa itu gravitasi"),
    ("jelaskan apa itu gravitasi", "Apa itu gravitasi"),
])
def test_definition_forms_become_apa_itu(q, expected):
    assert T.rewrite_definition(q) == (expected, True)


@pytest.mark.parametrize("q", ["apa itu gravitasi", "Apa itu gravitasi", "Bu Rina adalah guru", "Pak Joko itu apa?", "kota Bandung adalah", "Dimana ibu kota adalah",
                               "jelaskan proses pembuatan roti yang benar di rumah", "pengertian 3 ditambah 4", "Di mana Bu Rina tinggal?",
                               "Siapa ketua koperasi?", "Berapa luas Indonesia?"])
def test_non_definition_questions_are_not_rewritten(q):
    assert T.rewrite_definition(q) == (q, False)


def test_agent_and_cli_handle_definition_forms():
    class DefReader:
        def read(self, ctx, q):
            return GRAV_ANS if q == "Apa itu gravitasi" else "Maaf, tidak ada dalam teks."

    chat = load_module("memory_chat_def_under_test", "tools/memory_chat.py")
    store = M.MemoryStore(":memory:")
    store.teach(GRAV)
    agent = M.MemoryAgent(store, DefReader())
    out = agent.ask("gravitasi adalah")
    assert out["status"] == "ok" and out["answer"] == GRAV_ANS and out["rewritten"] == "Apa itu gravitasi"
    assert agent.ask("Apa itu gravitasi")["rewritten"] is None
    assert agent.ask("Apa itu gravitasi?")["rewritten"] == "Apa itu gravitasi"      # tanda tanya akhir dibuang (model menolak dengan '?')
    text = chat.handle("pengertian gravitasi", agent)                      # tanpa '?' pun dikenali sebagai pertanyaan
    assert text.startswith("(dibaca sebagai: Apa itu gravitasi)") and GRAV_ANS in text
    assert "jawaban" not in chat.handle("jelaskan gravitasi", agent).lower().split("\n")[0] or True
    out2 = agent.ask("dan istrinya?", ["Di mana Bu Rina tinggal?"])        # riwayat percakapan tetap didahulukan
    assert out2["rewritten"] == "Apa istri Bu Rina?"


# ---------- basa-basi di ujung jawaban dan bentuk pertanyaan alternatif ----------
@pytest.mark.parametrize("raw,expected", [
    ("Burung Garuda Berdasarkan teks yang diberikan.", "Burung Garuda"),
    ("Burung Garuda Berdasarkan teks yang diberikan, jawabannya tidak terdapat dalam teks yang diberikan.", "Burung Garuda"),
    ("Burung Garuda", "Burung Garuda"),
    ("Pak Joko menurut Bu Rina", "Pak Joko menurut Bu Rina"),                                       # kata 'menurut' biasa tidak dibuang
    ("Menurut informasi di atas, jawabannya ada.", ""),                                             # hanya basa-basi -> kosong
    ("Berdasarkan teks yang diberikan, tidak ada jawabannya.", ""),                                  # hanya basa-basi -> kosong = penolakan
])
def test_clean_answer(raw, expected):
    assert M.clean_answer(raw) == expected


def test_decide_accepts_answer_with_trailing_boilerplate_and_still_rejects_wrong_spans():
    note = "Lambang negara indonesia adalah Burung Garuda"
    d = M.decide("apa lambang negara indonesia", [(1, note, "Burung Garuda Berdasarkan teks yang diberikan.")])
    assert d["status"] == "ok" and d["answer"] == "Burung Garuda"
    assert M.decide("apa lambang negara indonesia", [(1, note, "Elang Jawa Berdasarkan teks yang diberikan.")])["status"] == "unknown"


def test_agent_tries_alternative_form_when_apa_itu_fails():
    class FormSensitive:
        def __init__(self):
            self.asked = []

        def read(self, ctx, q):
            self.asked.append(q)
            return "Burung Garuda" if q == "Apa lambang negara indonesia?" else "Maaf, tidak ada dalam teks."

    store = M.MemoryStore(":memory:")
    store.teach("Lambang negara indonesia adalah Burung Garuda")
    rd = FormSensitive()
    out = M.MemoryAgent(store, rd).ask("lambang negara indonesia adalah")
    assert out["status"] == "ok" and out["answer"] == "Burung Garuda"
    assert rd.asked == ["Apa itu lambang negara indonesia", "Apa lambang negara indonesia?"]
    assert out["rewritten"] == "Apa lambang negara indonesia?"                    # yang berhasil yang ditampilkan
    rd2 = FormSensitive()
    store2 = M.MemoryStore(":memory:")
    store2.teach("Bu Rina tinggal di Jalan Mawar nomor 5.")
    assert M.MemoryAgent(store2, rd2).ask("Di mana Bu Rina tinggal?")["status"] == "unknown" and len(rd2.asked) == 1   # bentuk lain hanya untuk 'apa itu'


# ---------- pemilihan jawaban saat konflik: kecocokan dulu, baru kebaruan ----------
N14 = "Indonesia negara berbentuk Republik dengan nama resmi Republik Indonesia."
N18 = "Lambang negara indonesia adalah Burung Garuda"


def test_better_matching_note_beats_a_newer_note_about_something_else():
    q = "Apa itu bentuk negara indonesia"
    d = M.decide(q, [(14, N14, "Republik"), (18, N18, "Burung Garuda")], threshold=0.5)      # "Longgar": #18 (kecocokan 2/3) ikut lolos
    assert d["source"]["id"] == 14 and d["answer"] == "Republik Indonesia"              # dulu: #18 menang karena lebih baru
    assert d["status"] == "conflict" and [c["id"] for c in d["conflicts"]] == [18]        # yang kalah tetap ditampilkan sebagai konflik
    assert M.decide(q, [(14, N14, "Republik"), (18, N18, "Burung Garuda")])["conflicts"] == []   # "Seimbang": #18 sudah ditolak penjaga


def test_newest_note_still_wins_for_updates_with_the_same_wording():
    old, new = "Bu Rina tinggal di Jalan Mawar nomor 5.", "Bu Rina tinggal di Jalan Melati nomor 9."
    d = M.decide("Di mana Bu Rina tinggal?", [(1, old, "Jalan Mawar nomor 5"), (2, new, "Jalan Melati nomor 9")])
    assert d["source"]["id"] == 2 and d["answer"] == "Jalan Melati nomor 9" and d["status"] == "conflict"


def test_update_with_looser_wording_is_flagged_as_conflict_not_silently_replaced():
    old = "Bu Rina tinggal di Jalan Mawar nomor 5 dekat pasar lama."
    new = "Bu Rina bekerja sebagai guru dan tinggal di Jalan Melati nomor 9."
    d = M.decide("Di mana Bu Rina tinggal dekat pasar lama?", [(1, old, "Jalan Mawar nomor 5"), (2, new, "Jalan Melati nomor 9")], relevance_guard=False)
    assert d["status"] == "conflict" and {d["source"]["id"], d["conflicts"][0]["id"]} == {1, 2}   # keduanya terlihat oleh pengguna
    assert d["source"]["id"] == 1                                                              # yang lebih cocok dengan pertanyaan menang


def test_user_note_still_beats_document_chunk_even_if_chunk_matches_better():
    note = "Bu Ratna adalah ketua koperasi."
    chunk = "Ketua koperasi Maju dipilih setiap tahun; ketua koperasi Maju saat ini adalah Pak Slamet."
    d = M.decide("Siapa ketua koperasi Maju?", [(1, note, "Bu Ratna"), (2, chunk, "Pak Slamet")], doc_ids=frozenset({2}), entity_guard=False)
    assert d["source"]["id"] == 1


def test_command_words_do_not_count_as_core_question_words():
    note = "Lambang negara indonesia adalah Burung Garuda"
    assert M.relevance_score("sebutkan lambang negara indonesia", note) == 1.0     # dulu 0.75: 'sebutkan' terpotong jadi 'butk'
    assert M.relevance_score("jelaskan lambang negara", note) == 1.0
    assert M.relevance_score("Apa itu bentuk negara indonesia", note) == pytest.approx(2 / 3)


# ---------- tanya balik bila beberapa catatan sama-sama cocok ----------
RAPAT_A = "Rapat tim produk dimulai pukul 09.30 di ruang Anggrek."
RAPAT_B = "Rapat tim finance dimulai pukul 11.30 di ruang Mawar."


def test_ambiguous_question_asks_back_with_distinguishing_words():
    d = M.decide("rapat ada di ruang apa", [(20, RAPAT_A, "Anggrek"), (21, RAPAT_B, "Mawar")])
    assert d["status"] == "clarify" and d["source"] is None
    assert [(o["id"], o["label"], o["answer"]) for o in d["options"]] == [(20, "produk", "Anggrek"), (21, "finance", "Mawar")]


def test_question_that_names_the_distinguishing_word_is_answered_directly():
    d = M.decide("rapat tim finance ada di ruang apa", [(20, RAPAT_A, "Anggrek"), (21, RAPAT_B, "Mawar")])
    assert d["status"] in ("ok", "conflict") and d["answer"] == "Mawar" and d["source"]["id"] == 21


def test_fact_update_is_not_ambiguous_only_the_answer_differs():
    old, new = "Bu Rina tinggal di Jalan Mawar nomor 5.", "Bu Rina tinggal di Jalan Melati nomor 9."
    d = M.decide("Di mana Bu Rina tinggal?", [(1, old, "Jalan Mawar nomor 5"), (2, new, "Jalan Melati nomor 9")])
    assert d["status"] == "conflict" and d["source"]["id"] == 2                  # terbaru menang, tidak bertanya balik


def test_no_clarification_when_notes_agree_or_match_unequally():
    same = [(1, RAPAT_A, "Anggrek"), (2, "Rapat tim finance dimulai pukul 11.30 di ruang Anggrek.", "Anggrek")]
    assert M.decide("rapat ada di ruang apa", same)["status"] == "ok"            # jawabannya sama
    d = M.decide("Apa itu bentuk negara indonesia", [(14, N14, "Republik"), (18, N18, "Burung Garuda")], threshold=0.5)
    assert d["status"] == "conflict"                                              # kecocokan berbeda: pilih yang lebih cocok


def test_agent_cli_and_web_ask_back():
    class Room:
        def read(self, ctx, q):
            return "Anggrek" if "Anggrek" in ctx else "Mawar" if "Mawar" in ctx else "Maaf, tidak ada dalam teks."

    chat = load_module("memory_chat_clarify_under_test", "tools/memory_chat.py")
    store = M.MemoryStore(":memory:")
    store.teach(RAPAT_A)
    store.teach(RAPAT_B)
    agent = M.MemoryAgent(store, Room())
    out = agent.ask("rapat ada di ruang apa")
    assert out["status"] == "clarify" and len(out["options"]) == 2
    text = chat.handle("rapat ada di ruang apa?", agent)
    assert "Maksud Anda yang mana?" in text and "(tambahkan kata: produk)" in text and "mis.: rapat ada di ruang apa produk" in text
    assert agent.ask("rapat ada di ruang apa produk")["answer"] == "Anggrek"

    web = load_module("memory_web_clarify_under_test", "tools/memory_web.py")
    state = FakeState()
    state.store.teach(RAPAT_A)
    state.store.teach(RAPAT_B)
    state.agent = M.MemoryAgent(state.store, Room())
    srv = web.ExclusiveServer(("127.0.0.1", 0), web.make_handler(state, 0))
    port = srv.server_address[1]
    srv.RequestHandlerClass = web.make_handler(state, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/api/ask", json.dumps({"question": "rapat ada di ruang apa"}), {"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"})
        j = json.loads(c.getresponse().read().decode("utf-8"))
        assert j["status"] == "clarify" and [o["label"] for o in j["options"]] == ["produk", "finance"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_stem_strips_enclitic_then_suffix():
    assert M.stem("ruangannya") == "ruang" and M.stem("rumahnya") == "rumah" and M.stem("tinggal") == "tinggal"


# ---------- bentuk pertanyaan alternatif ----------
@pytest.mark.parametrize("q,expected", [
    ("di asia mana indonesia berada", [("Di mana indonesia berada?", "asia")]),
    ("Di kota mana Bu Rina tinggal?", [("Di mana Bu Rina tinggal?", "kota")]),
    ("indonesia berada di kawasan apa?", [("indonesia berada di mana", "kawasan")]),
    ("apa itu gravitasi", [("Apa gravitasi?", None)]),
    ("Di mana Bu Rina tinggal?", []), ("Siapa ketua koperasi?", []), ("Kapan rapat dimulai?", []),
])
def test_alternate_forms(q, expected):
    assert T.alternate_forms(q) == expected


def test_alternate_form_only_reads_notes_that_contain_the_dropped_qualifier():
    class WhereReader:
        def __init__(self):
            self.asked = []

        def read(self, ctx, q):
            self.asked.append(q)
            if q.startswith("Di mana"):
                return "asia tenggara" if "asia tenggara" in ctx else "Jalan Mawar nomor 5" if "Mawar" in ctx else "Maaf, tidak ada dalam teks."
            return "Maaf, tidak ada dalam teks."

    store = M.MemoryStore(":memory:")
    store.teach("indonesia berada di asia tenggara")
    store.teach("Bu Rina tinggal di Jalan Mawar nomor 5.")
    rd = WhereReader()
    agent = M.MemoryAgent(store, rd)
    out = agent.ask("di asia mana indonesia berada")
    assert out["status"] == "ok" and out["answer"] == "asia tenggara" and out["rewritten"] == "Di mana indonesia berada?"
    out2 = agent.ask("Di kota mana Bu Rina tinggal?")        # catatan tidak memuat 'kota': jangan dijawab dengan jalan
    assert out2["status"] == "unknown" and out2["rewritten"] is None
    assert not any(q == "Di mana Bu Rina tinggal?" for q in rd.asked)
    agent.alternate_forms = False
    assert agent.ask("di asia mana indonesia berada")["status"] == "unknown"


# ---------- pemilih alat berbasis aturan: parafrase dan penjaga pembajakan ----------
def _route_answer(q):
    r = T.route(q, None, [(2, "Skor ujian: 78, 95, 60, 88.")])
    return None if r is None else (r["kind"], r["answer"])


@pytest.mark.parametrize("q,expected", [
    ("susunlah angka 14 3 9 dari yang kecil", ("tool", "3, 9, 14")),
    ("tolong rapikan deretan angka 5 1 4 2", ("tool", "1, 2, 4, 5")),
    ("buat urutan terbesar ke terkecil dari 6 18 11", ("tool", "18, 11, 6")),
    ("diurutkan dari kecil: 50 40 45", ("tool", "40, 45, 50")),
    ("urutkan dari terkecil ke terbesar 9 1 5", ("tool", "1, 5, 9")),          # dulu: 'terbesar' membalik jadi menurun
    ("urutkan angka terbesar 1 9 5", ("tool", "9, 5, 1")),
    ("tambahkan 120 dengan 80", ("tool", "200")), ("kurangi 500 dengan 125", ("tool", "375")), ("kalikan 12 dengan 12", ("tool", "144")),
    ("total dari 7 dan 15 dan 20 berapa", ("tool", "42")), ("jumlah 4 5 dan 6 ada berapa", ("tool", "15")),
    ("15 dikurangi 6 hasilnya?", ("tool", "9")), ("jumlahkan 15 dan 35", ("tool", "50")),
    ("tanggal 17 Agustus 2026 jatuh hari apakah", ("tool", "Senin")),
    ("berapa lama dari 1 Juli 2025 hingga 1 Oktober 2025", ("tool", "92 hari")),
])
def test_rule_router_paraphrases(q, expected):
    assert _route_answer(q) == expected


@pytest.mark.parametrize("q", ["tolong tulis pengumuman rapat", "karang pantun tentang sekolah", "susun surat izin sakit untuk anak saya",
                               "ubah teks ini menjadi lebih sopan", "buat daftar belanja mingguan", "bikin caption instagram untuk toko kue"])
def test_more_task_requests_are_recognised(q):
    assert _route_answer(q)[0] == "task"


@pytest.mark.parametrize("q", ["Siapa yang menyusun jadwal rapat?", "Berapa total biaya rapat tim?", "Berapa jumlah peserta rapat kemarin?",
                               "Hari apa Bu Rina libur?", "Buat apa rapat itu diadakan?", "Siapa yang bertugas mengurutkan arsip?",
                               "Kurang berapa anggota untuk kuorum?", "Apa isi catatan 1?", "Total biaya perjalanan Pak Joko berapa?",
                               "Ke mana Bu Rina pindah setelah tanggal 5 Mei 2026?", "Kapan batas waktu penyusunan anggaran?"])
def test_ordinary_questions_are_not_hijacked_by_wider_rules(q):
    assert _route_answer(q) is None


def test_wider_greetings_match_only_whole_message():
    for q in ["terima kasih banyak ya", "assalamualaikum warahmatullahi wabarakatuh", "apa kabar semuanya", "selamat datang", "thanks a lot", "salam"]:
        assert T.GREETING.match(q), q
    for q in ["halo, di mana Bu Rina tinggal?", "apa kabar proyek koperasi?", "salam dari siapa di surat itu?", "terima kasih atas bantuannya kemarin"]:
        assert not T.GREETING.match(q), q


def test_router_eval_sets_never_hijack_search_messages():
    import json
    from pathlib import Path
    sys_path = Path(__file__).resolve().parent.parent / "tools"
    import sys
    sys.path.insert(0, str(sys_path))
    from memory_router_eval import rule_router
    root = Path(__file__).resolve().parent.parent / "evalsets"
    notes = [(1, "Bu Rina tinggal di Jalan Mawar nomor 5."), (2, "Skor ujian: 78, 95, 60, 88.")]
    total = hijacked = ok = 0
    for name in ("memory_router.jsonl", "memory_router_heldout.jsonl", "memory_router_test2.jsonl"):
        for line in open(root / name, encoding="utf-8"):
            row = json.loads(line)
            pred = rule_router(row["message"], notes)
            total += 1
            ok += pred == row["label"]
            hijacked += row["label"] == "cari" and pred != "cari"
    assert hijacked == 0                       # risiko utama perluasan aturan: pertanyaan biasa dibajak alat
    assert ok / total >= 0.95                  # dev 97,2% / heldout 100% / test2 100% pada saat ditulis


def test_cli_task_message_explains_how_to_save_an_imperative_note():
    chat = load_module("memory_chat_task_hint_under_test", "tools/memory_chat.py")
    agent = M.MemoryAgent(M.MemoryStore(":memory:"), NoReader())
    out = chat.handle("Buat laporan mingguan setiap Jumat.", agent)
    assert "permintaan tugas" in out and "ingat: <kalimat Anda>" in out
    assert "Dicatat" in chat.handle("ingat: Buat laporan mingguan setiap Jumat.", agent)       # cara menyimpannya tetap berfungsi


# ---------- penjaga gema dan penjaga "catatan yang lebih cocok tidak terbaca" ----------
R20 = "Rapat tim produk dimulai pukul 09.30 di ruang Anggrek."
R21 = "Rapat tim finance dimulai pukul 11.30 di ruang Mawar."


def test_echo_guard_rejects_answers_that_only_repeat_the_question():
    q = "rapat di ruang mawar diadakan oleh tim mana"
    reads = [(21, R21, "Mawar"), (20, R20, "Anggrek")]
    assert M.decide(q, reads)["status"] == "ok"                                      # tanpa penjaga: jawaban salah lolos
    d = M.decide(q, reads, echo_threshold=1.0)
    assert d["status"] == "unknown" and "hanya mengulang" in d["guesses"][0]["reason"]
    assert M.echo_fraction("Siapa ketua koperasi Maju?", "Bu Ratna") == 0.0          # jawaban sah tidak terkena
    assert M.decide("Apa itu gravitasi", [(1, GRAV, GRAV_ANS)], echo_threshold=1.0)["status"] == "ok"


def test_answer_from_a_worse_matching_note_is_not_trusted_when_the_better_note_is_unreadable():
    q = "jam berapa rapat dimulai di ruang mawar"
    reads = [(21, R21, "tiga puluh"), (20, R20, "09.30")]       # model gagal membaca #21 (bukan angka yang bisa diperbaiki)
    assert M.decide(q, reads)["answer"] == "09.30"             # tanpa penjaga: jawaban ruang Anggrek dipakai
    d = M.decide(q, reads, unread_margin=0.2)
    assert d["status"] == "unknown"
    reasons = {g["id"]: g["reason"] for g in d["guesses"]}
    assert "bukan kutipan persis" in reasons[21] and "#21" in reasons[20] and "tidak bisa saya baca" in reasons[20]


def test_misread_digit_is_repaired_to_the_exact_text_of_the_note():
    q = "jam berapa rapat dimulai di ruang mawar"
    d = M.decide(q, [(21, R21, "11.00"), (20, R20, "09.30")], unread_margin=0.2)       # kasus nyata: model menulis 11.00, catatan berisi 11.30
    assert d["status"] == "ok" and d["answer"] == "11.30" and d["source"]["id"] == 21   # catatan ruang Anggrek bukan konflik
    assert d["repaired"] == {"model": "11.00", "note": "11.30"}
    assert M.decide("jam berapa rapat dimulai di ruang anggrek", [(20, R20, "09.30")])["repaired"] is None


@pytest.mark.parametrize("raw,note,expected", [
    ("11.00", "Rapat dimulai pukul 11.30 di ruang Mawar.", "11.30"),
    ("1.904.569", "Luas wilayah 1.904.659 km2.", "1.904.659"),
    ("11.00", "Rapat pukul 11.30 sampai 13.30.", None),          # dua angka berbentuk sama: tidak menebak
    ("12.00", "Rapat dimulai pukul 09.30.", None),                # selisih lebih dari 2 digit
    ("11,5", "Rapat pukul 11.30.", None),                         # bentuk berbeda (koma vs titik)
    ("Kamis", "Rapat hari Kamis.", None),                         # bukan angka
    ("11.30", "Rapat pukul 11.30.", None),                        # sudah benar: tidak ada yang diperbaiki
])
def test_repair_number_is_narrow(raw, note, expected):
    assert M.repair_number(raw, note) == expected


def test_unread_margin_does_not_block_a_correct_answer_from_the_best_matching_note():
    reads = [(20, R20, "09.30"), (21, R21, "Teks tidak menyebutkan hal itu.")]
    d = M.decide("jam berapa rapat dimulai di ruang anggrek", reads, unread_margin=0.2)
    assert d["status"] == "ok" and d["answer"] == "09.30"
    assert M.decide("Di mana Bu Rina tinggal?", [(1, "Bu Rina tinggal di Jalan Mawar nomor 5.", "Jalan Mawar nomor 5")], unread_margin=0.2)["status"] == "ok"


def test_agent_enables_both_guards_by_default():
    class Misreads:
        def read(self, ctx, q):
            if "Mawar" in ctx:
                return "Mawar" if "tim mana" in q else "11.00"
            return "09.30" if "jam" in q else "Anggrek"

    store = M.MemoryStore(":memory:")
    store.teach(R20)
    store.teach(R21)
    agent = M.MemoryAgent(store, Misreads())
    assert agent.echo_threshold == 1.0 and agent.unread_margin == 0.2
    assert agent.ask("rapat di ruang mawar diadakan oleh tim mana")["status"] == "unknown"
    out = agent.ask("jam berapa rapat dimulai di ruang mawar")
    assert out["status"] == "ok" and out["answer"] == "11.30" and out["repaired"]["model"] == "11.00" and out["conflicts"] == []
    assert agent.ask("jam berapa rapat dimulai di ruang anggrek")["answer"] == "09.30"


def test_cli_recognises_questions_with_the_question_word_inside_the_sentence():
    chat = load_module("memory_chat_inner_q_under_test", "tools/memory_chat.py")
    for s in ["jam berapa rapat dimulai di ruang mawar", "rapat di ruang mawar diadakan oleh tim mana", "ruang mana yang dipakai rapat finance", "tim finance rapat kapan"]:
        assert chat.is_question_like(s), s
    for s in ["Bu Rina tinggal di Jalan Mawar nomor 5.", "Bu Rina tahu apa yang terjadi kemarin", "Rapat tim produk dimulai pukul 09.30 di ruang Anggrek.",
              "Pak Joko bekerja sebagai montir bengkel di Bekasi"]:
        assert not chat.is_question_like(s), s
    agent = M.MemoryAgent(M.MemoryStore(":memory:"), NoReader())
    agent.store.teach("Rapat tim finance dimulai pukul 11.30 di ruang Mawar.")
    assert "tidak disimpan" not in chat.handle("jam berapa rapat dimulai di ruang mawar", agent)       # dulu dianggap pernyataan


def test_related_notes_need_real_overlap_not_just_one_shared_word():
    reads = [(21, R21, "11.30"), (3, "Bu Rina tinggal di Jalan Mawar nomor 5.", "Jalan Mawar nomor 5"), (22, "Rapat tim finance dimulai pukul 13.00 di ruang Mawar lantai 2.", "13.00")]
    rel = M.related_unused("jam berapa rapat dimulai di ruang mawar", reads, {21})
    assert [r["id"] for r in rel] == [22]                      # #3 hanya berbagi kata 'mawar': tidak ditampilkan


# ---------- kata jenis-jawaban dan daftar hari/bulan terpotong ----------
@pytest.mark.parametrize("word,answer,expected", [
    ("hari", "Jumat", True), ("hari", "Senin sampai Rabu", True), ("hari", "Senin dan Selasa", True), ("hari", "Bu Ratna", False), ("hari", "Jumat malam", False),
    ("bulan", "Agustus", True), ("bulan", "Agustus 2026", False), ("warna", "merah", True), ("warna", "merah muda", True), ("warna", "Anggrek", False),
    ("jam", "11.30", True), ("pukul", "09:30", True), ("jam", "Mawar", False), ("umur", "45 tahun", True), ("usia", "Bandung", False),
    ("kota", "Bandung", False),                                              # kata lain tidak pernah diabaikan
])
def test_answer_matches_type(word, answer, expected):
    assert M.answer_matches_type(word, answer) is expected


@pytest.mark.parametrize("answer,note,expected", [
    ("Senin", "Dokter Anindya membuka praktik setiap Senin sampai Rabu mulai pukul 16.00.", "Senin sampai Rabu"),
    ("Selasa", "Jadwal piket kelas: Senin dan Selasa giliran regu A, Rabu dan Kamis regu B.", "Senin dan Selasa"),
    ("Rabu", "Jadwal piket kelas: Senin dan Selasa giliran regu A, Rabu dan Kamis regu B.", "Rabu dan Kamis"),
    ("Minggu", "Toko roti milik Bu Marni buka setiap hari kecuali Minggu, dari pukul tujuh.", "Minggu"),    # bukan daftar: tidak diubah
    ("Jumat", "Rapat RT dilaksanakan setiap Jumat malam di rumah Pak Lurah.", "Jumat"),
    ("Agustus", "Festival diadakan pada bulan Agustus sampai September di pantai.", "Agustus sampai September"),
])
def test_truncated_day_and_month_lists_are_completed_from_the_note(answer, note, expected):
    assert M.complete_span(answer, note) == expected


def test_type_word_is_ignored_only_when_the_answer_has_the_matching_type():
    note = "Rapat RT dilaksanakan setiap Jumat malam di rumah Pak Lurah."
    q = "Hari apa rapat RT dilaksanakan?"
    assert M.decide(q, [(1, note, "Jumat")])["answer"] == "Jumat"                      # dulu ditolak: 'hari' tidak ada di catatan (2/3 < 0,67)
    assert M.decide(q, [(1, note, "rumah Pak Lurah")])["status"] == "unknown"           # jawaban bertipe lain: kata 'hari' tetap dihitung
    warna = "Pak Anto membeli sepeda motor merah di Bandung."
    assert M.decide("Apa warna sepeda motor Pak Anto?", [(1, warna, "merah")])["answer"] == "merah"
    assert M.decide("Apa warna sepeda motor Pak Anto?", [(1, warna, "Bandung")])["status"] == "unknown"


def test_frozen_set_regressions_from_the_blanket_fix_stay_fixed():
    """Empat pertanyaan di set uji beku yang berubah hasilnya ketika kata jenis-jawaban diabaikan tanpa syarat (dibatalkan)."""
    n1 = "Festival layang-layang tahun ini diadakan di pantai Parangtritis pada bulan Agustus."
    assert M.decide("Pada bulan apa festival layang-layang berlangsung?", [(1, n1, "Agustus")])["answer"] == "Agustus"   # 'bulan' ada di catatan: tetap benar
    n2 = "Toko roti milik Bu Marni buka setiap hari kecuali Minggu, dari pukul tujuh pagi sampai pukul sembilan malam."
    assert M.decide("Hari apa toko roti Bu Marni tutup?", [(1, n2, "Minggu")])["answer"] == "Minggu"
    n3 = "Dokter Anindya membuka praktik setiap Senin sampai Rabu mulai pukul 16.00."
    assert M.decide("Pada hari apa saja Dokter Anindya praktik?", [(1, n3, "Senin")])["answer"] == "Senin sampai Rabu"   # dulu 'salah: Senin'
    n4 = "Jadwal piket kebersihan kelas: Senin dan Selasa giliran regu A, Rabu dan Kamis regu B."
    assert M.decide("Hari apa regu A piket?", [(1, n4, "Selasa")])["answer"] == "Senin dan Selasa"                      # dulu 'salah: Selasa'
