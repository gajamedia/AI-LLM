"""Tes aturan "jika ... jawab ...", sapaan di agen/web, kata bilangan, dan saran tabel (tanpa memuat model)."""
import pytest

import memory_agent as M
import memory_tools as T
from conftest import load_module


class NoReader:
    def read(self, *a, **k):
        raise AssertionError("model tidak boleh dipanggil")


def agent_with(*notes):
    store = M.MemoryStore(":memory:")
    for n in notes:
        store.teach(n)
    return M.MemoryAgent(store, NoReader())


# ---------- aturan ----------
@pytest.mark.parametrize("text,expected", [
    ('jika ada yang menyapa "halo" jawab saja "halo"', ("halo", "halo")),
    ('Kalau ditanya “siapa kamu” jawab “Saya memori pribadi Anda.”', ("siapa kamu", "Saya memori pribadi Anda.")),
    ('jika seseorang bilang "terima kasih" maka jawablah "sama-sama"', ("terima kasih", "sama-sama")),
    ('apabila ada yang menulis "ping", jawab "pong".', ("ping", "pong")),
])
def test_rule_notes_are_parsed(text, expected):
    assert M.parse_rule(text) == expected


@pytest.mark.parametrize("text", [
    "Bu Rina tinggal di Jalan Mawar nomor 5.",
    'Pak Joko pernah bilang "halo" kepada Bu Rina.',     # ada kutipan tetapi bukan aturan
    "jika hujan, rapat dipindah ke ruang Anggrek.",       # "jika" tanpa kutipan dan tanpa "jawab"
    "jika ada yang menyapa halo jawab saja halo",          # tanpa tanda kutip: ambigu, tidak dianggap aturan
    'jawab "halo" jika ada yang menyapa',                  # urutan terbalik
])
def test_ordinary_notes_are_not_rules(text):
    assert M.parse_rule(text) is None


def test_rule_runs_only_on_exact_trigger_and_without_the_model():
    a = agent_with('jika ada yang menyapa "halo" jawab saja "halo"', "Bu Rina tinggal di Jalan Mawar nomor 5.")
    for q in ("HALO", "halo", "Halo!", "  halo  ", "halo?"):
        out = a.ask(q)
        assert out["status"] == "rule" and out["answer"] == "halo" and out["source"]["id"] == 1, q


def test_rule_does_not_fire_on_similar_or_normal_questions():
    a = agent_with('jika ada yang menyapa "halo" jawab saja "halo"')
    assert M.match_rule(a.store, "Halo apa kabar") is None
    assert M.match_rule(a.store, "ha lo") is None
    assert M.match_rule(a.store, "Di mana Bu Rina tinggal?") is None
    assert M.match_rule(a.store, "") is None


def test_newest_matching_rule_wins_and_forgotten_rules_stop_working():
    a = agent_with('jika ada yang menyapa "halo" jawab "Halo lama"', 'jika ada yang menyapa "halo" jawab "Halo baru"')
    assert a.ask("halo")["answer"] == "Halo baru"
    a.store.forget(2)
    assert a.ask("halo")["answer"] == "Halo lama"
    a.store.forget(1)
    assert a.ask("halo")["status"] == "greeting"  # tanpa aturan, kembali ke sapaan bawaan


def test_user_rule_overrides_builtin_greeting():
    a = agent_with('jika ada yang menyapa "halo" jawab "Selamat datang, Bos!"')
    assert a.ask("halo") ["answer"] == "Selamat datang, Bos!"
    assert a.ask("selamat pagi")["status"] == "greeting"  # sapaan lain tetap bawaan


# ---------- cakupan sapaan ----------
@pytest.mark.parametrize("q", ["halo", "Halo apa kabar", "selamat pagi semua", "Hai kak", "terima kasih banyak", "assalamualaikum", "apa kabar?"])
def test_greeting_variants_are_recognised(q):
    assert T.GREETING.match(q.strip())


@pytest.mark.parametrize("q", ["halo di mana Bu Rina tinggal", "halo Bu Rina tinggal di Jalan Mawar nomor 5", "hai kamu siapa",
                              "selamat pagi rapat dimulai pukul sembilan"])
def test_greeting_does_not_swallow_real_content(q):
    assert not T.GREETING.match(q.strip())


# ---------- sapaan di agen (dipakai web) ----------
def test_greeting_in_agent_without_model_and_with_note_count():
    empty = agent_with().ask("Halo")
    assert empty["status"] == "greeting" and "ingat:" in empty["answer"]
    two = agent_with("Bu Rina tinggal di Jalan Mawar nomor 5.", "Pak Joko bekerja sebagai montir.").ask("selamat pagi")
    assert two["status"] == "greeting" and "2 catatan" in two["answer"]


# ---------- kata bilangan ----------
@pytest.mark.parametrize("words,digits", [
    ("satu", "1"), ("sepuluh", "10"), ("sebelas", "11"), ("dua belas", "12"), ("dua puluh lima", "25"),
    ("seratus tiga belas", "113"), ("seratus dua puluh tiga", "123"), ("dua ribu lima ratus", "2500"),
    ("seribu dua ratus tiga puluh empat", "1234"), ("tiga satu dua", "3 1 2"), ("sembilan puluh sembilan", "99"),
])
def test_number_words(words, digits):
    assert T.words_to_digits(words) == digits


@pytest.mark.parametrize("q,expected", [
    ("berapa satu ditambah satu?", "2"),
    ("berapa dua puluh lima kali dua?", "50"),
    ("seratus dikurangi tujuh puluh", "30"),
    ("hitung sepuluh dibagi empat", "2,5"),
    ("tiga ditambah empat", "7"),
    ("urutkan tiga, satu, dua", "1, 2, 3"),
    ("urutkan lima belas dua puluh tiga satu", "1, 15, 23"),
])
def test_number_words_in_tools(q, expected):
    assert T.route(q)["answer"] == expected


@pytest.mark.parametrize("q", [
    "Berapa harga satu kotak donat?",
    "Berapa harga satu kotak donat ditambah dua kotak roti?",
    "Siapa yang lahir tanggal satu Januari?",
    "Rapat dimulai pukul sembilan di ruang satu.",
    "Bu Rina punya dua anak dan satu cucu.",
    "satu-satunya anak Pak Joko bernama Budi",
])
def test_number_words_do_not_hijack_normal_sentences(q):
    assert T.route(q) is None


# ---------- saran tabel/rumus ----------
def test_table_notes_are_detected():
    assert T.looks_like_table("Penjumlahan: 1+1=2 1+2=3 1+3=4 1+4=5")
    assert T.looks_like_table("a = 1, b = 2, c = 3")
    assert not T.looks_like_table("Bu Rina tinggal di Jalan Mawar nomor 5.")
    assert not T.looks_like_table("Harga donat 45 ribu.")


def test_web_hints_for_tables_and_rules():
    web = load_module("memory_web_rules_under_test", "tools/memory_web.py")
    assert T.TABLE_HINT in web.note_hints("Penjumlahan: 1+1=2 1+2=3 1+3=4 1+4=5")
    assert web.note_hints('jika ada yang menyapa "halo" jawab saja "halo"') == []  # aturan: tanpa saran gaya
    assert web.note_hints("Bu Rina tinggal di Jalan Mawar nomor 5.") == []


# ---------- CLI ----------
def test_cli_rule_and_teach_messages():
    chat = load_module("memory_chat_rules_under_test", "tools/memory_chat.py")
    a = agent_with()
    out = chat.handle('ingat: jika ada yang menyapa "halo" jawab saja "halo"', a)
    assert "Dikenali sebagai aturan" in out and '"halo"' in out
    out = chat.handle("halo", a)
    assert out.startswith("halo") and "aturan dari catatan [1]" in out
    out = chat.handle("ingat: Penjumlahan: 1+1=2 1+2=3 1+3=4", a)
    assert out.startswith("Dicatat [2].") and "Tabel atau rumus" in out
    assert "Halo!" in chat.handle("selamat pagi", a)  # sapaan lain tetap bawaan
