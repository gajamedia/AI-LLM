"""Tes alat deterministik Memori dan integrasinya (tanpa memuat model)."""
import pytest

import memory_tools as T
from conftest import load_module


def ans(q):
    r = T.route(q)
    assert r is not None, f"tidak dikenali: {q!r}"
    return r["answer"]


# ---------- pengurut angka ----------
@pytest.mark.parametrize("q,expected", [
    ("urutkan deret angka ini (5-4-3-9)", "3, 4, 5, 9"),            # kasus dari pengguna
    ("urutkan 5, 4, 3, 9", "3, 4, 5, 9"),
    ("urutkan 5, 4, 3, 9 dari besar ke kecil", "9, 5, 4, 3"),
    ("Urutkan angka 10 2 33 4", "2, 4, 10, 33"),                     # urut numerik, bukan teks
    ("urutkan -3, 7, 0, -10", "-10, -3, 0, 7"),
    ("urutkan 1.000 250 30", "30, 250, 1000"),                       # titik ribuan
    ("urutkan 5,4,3,9", "3, 4, 5, 9"),                               # koma tanpa spasi = daftar
    ("urutkan angka terbesar 1 9 5", "9, 5, 1"),
])
def test_sort(q, expected):
    r = T.route(q)
    assert r["kind"] == "tool" and r["tool"] == "pengurut angka"
    assert r["answer"] == expected


def test_sort_decimals_use_unambiguous_separator():
    assert ans("urutkan 3,5 1,2 2,8") == "1,2; 2,8; 3,5"


def test_sort_without_numbers_is_a_task_message_not_a_guess():
    r = T.route("urutkan nama-nama ini")
    assert r["kind"] == "task"


# ---------- kalkulator ----------
@pytest.mark.parametrize("q,expected", [
    ("berapa 3 + 4?", "7"),
    ("hitung 12 x 5", "60"),
    ("100 dibagi 4", "25"),
    ("2 pangkat 10", "1024"),
    ("berapa (3+4)*2", "14"),
    ("5 dikali 6", "30"),
    ("3+4", "7"),
    ("hitung 7 / 2", "3,5"),
    ("berapa 10 dikurangi 25", "-15"),
    ("hitung 1.000 + 250", "1250"),
])
def test_calc(q, expected):
    assert ans(q) == expected


def test_calc_division_by_zero_and_huge_power_are_safe():
    assert "dibagi nol" in ans("hitung 10 / 0")
    r = T.route("hitung 9 pangkat 99999")  # ditolak dengan pesan yang benar, tanpa menghitung bilangan raksasa
    assert r["kind"] == "tool" and "Terlalu besar" in r["answer"]


# ---------- tanggal ----------
def test_date_tools():
    assert ans("hari apa 17 Agustus 2026?") == "Senin"
    assert ans("berapa hari antara 1 Januari 2026 dan 1 Maret 2026?") == "59 hari"
    assert T.route("hari apa 31 Februari 2026?") is None  # tanggal tidak valid


# ---------- permintaan tugas ----------
@pytest.mark.parametrize("q", ["terjemahkan kalimat ini ke bahasa Inggris", "tuliskan puisi tentang hujan",
                               "tolong ringkaskan teks ini", "buatkan saya cerita"])
def test_task_requests_get_clear_message(q):
    r = T.route(q)
    assert r["kind"] == "task" and "bukan pertanyaan tentang catatan" in r["answer"]


# ---------- tidak boleh membajak pertanyaan/pernyataan biasa ----------
@pytest.mark.parametrize("q", [
    "Siapa yang lahir tanggal 12-03-1990?",
    "Nomor telepon Pak Joko 0812-345-678 berapa?",
    "Di mana Bu Rina tinggal?",
    "Berapa harga satu kotak donat?",
    "Berapa harga 3 kotak donat ditambah 2 kotak roti?",
    "Rapat dimulai pukul 09.30 di ruang Anggrek.",
    "Bu Rina lahir pada 17 Agustus 1990.",
    "Siapa ketua tim 5-4?",
    "urutan angka",
    "Apa pekerjaan Pak Joko?",
])
def test_normal_questions_and_statements_are_not_hijacked(q):
    assert T.route(q) is None


# ---------- integrasi ----------
class FakeAgent:
    def __init__(self):
        from memory_agent import MemoryStore
        self.store = MemoryStore(":memory:")
        self.asked = []

    def ask(self, q, history=None):
        self.asked.append(q)
        return {"status": "unknown", "answer": "belum tahu", "source": None, "conflicts": [], "related": []}


@pytest.fixture(scope="module")
def chat():
    return load_module("memory_chat_tools_under_test", "tools/memory_chat.py")


def test_cli_runs_tools_even_when_memory_is_empty_and_never_offers_to_save_them(chat):
    agent = FakeAgent()
    saved = []
    out = chat.handle("urutkan deret angka ini (5-4-3-9)", agent, confirm=lambda p: saved.append(p) or "y")
    assert out.startswith("3, 4, 5, 9") and "alat" in out and "bukan dari catatan" in out
    assert saved == [] and agent.store.list() == [] and agent.asked == []


def test_cli_task_request_is_not_treated_as_a_statement(chat):
    agent = FakeAgent()
    out = chat.handle("terjemahkan kalimat ini ke bahasa Inggris", agent, confirm=lambda p: "y")
    assert "permintaan tugas" in out and agent.store.list() == []


def test_agent_ask_routes_to_tools_without_the_model():
    from memory_agent import MemoryAgent, MemoryStore

    class NoReader:
        def read(self, *a, **k):
            raise AssertionError("model tidak boleh dipanggil untuk alat")

    a = MemoryAgent(MemoryStore(":memory:"), NoReader())
    out = a.ask("urutkan 5, 4, 3, 9")
    assert out["status"] == "tool" and out["answer"] == "3, 4, 5, 9" and out["source"] is None
    assert a.ask("tuliskan puisi")["status"] == "task"
