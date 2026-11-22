"""Tes perutean masukan CLI memori (tanpa memuat model)."""
import pytest

from conftest import load_module


@pytest.fixture(scope="module")
def chat():
    return load_module("memory_chat_under_test", "tools/memory_chat.py")


class FakeAgent:
    """Pengganti MemoryAgent: penyimpan SQLite sungguhan, tetapi ask() hanya mencatat panggilan."""

    def __init__(self, store):
        self.store = store
        self.asked = []

    def ask(self, q, history=None):
        self.asked.append(q)
        return {"status": "unknown", "answer": "Saya belum tahu; belum ada catatan yang menjawabnya.",
                "source": None, "conflicts": [], "related": []}


@pytest.fixture()
def agent():
    from memory_agent import MemoryStore
    return FakeAgent(MemoryStore(":memory:"))


def test_greeting_is_friendly_and_does_not_query_model(chat, agent):
    out = chat.handle("halo", agent)
    assert "Halo" in out and "ingat:" in out  # memori kosong: memberi contoh cara mulai
    assert agent.asked == []
    assert "Halo" in chat.handle("Selamat pagi", agent)
    assert "Halo" in chat.handle("terima kasih!", agent)


def test_question_on_empty_memory_says_memory_is_empty(chat, agent):
    out = chat.handle("presiden indonesia sekarang?", agent)
    assert "Memori masih kosong" in out
    assert agent.asked == []  # model tidak dipanggil


def test_question_with_question_word_but_no_mark_is_treated_as_question(chat, agent):
    agent.store.teach("Bu Rina tinggal di Jalan Mawar nomor 5.")
    out = chat.handle("siapa presiden indonesia sekarang", agent)
    assert agent.asked == ["siapa presiden indonesia sekarang"]
    assert "belum tahu" in out and "ingat:" in out  # petunjuk mengajari
    chat.handle("di mana Bu Rina tinggal", agent)
    assert len(agent.asked) == 2


def test_statement_offers_to_save_and_respects_answer(chat, agent):
    asked = []

    def yes(prompt):
        asked.append(prompt)
        return "y"

    out = chat.handle("Bu Rina tinggal di Jalan Mawar nomor 5.", agent, confirm=yes)
    assert out.startswith("Dicatat [") and asked and "Simpan sebagai catatan" in asked[0]
    assert [t for _, t in agent.store.list()] == ["Bu Rina tinggal di Jalan Mawar nomor 5."]

    out = chat.handle("Pak Joko bekerja sebagai montir bengkel.", agent, confirm=lambda p: "n")
    assert "Tidak disimpan" in out
    assert len(agent.store.list()) == 1


def test_statement_in_noninteractive_mode_is_not_saved_but_explains(chat, agent):
    out = chat.handle("Bu Rina tinggal di Jalan Mawar nomor 5.", agent, confirm=None)
    assert "tidak disimpan" in out and "ingat: Bu Rina tinggal" in out
    assert agent.store.list() == []


def test_too_short_input_gets_guidance_not_a_bare_error(chat, agent):
    for text in ("indonesia", "ok sip", "presiden indonesia sekarang"):
        out = chat.handle(text, agent)
        assert "terlalu pendek" in out and "ingat:" in out
    assert agent.store.list() == []


def test_x_adalah_is_read_as_a_definition_question_like_in_the_web_app(chat, agent):
    out = chat.handle("indonesia adalah", agent)        # dulu 'terlalu pendek'; kini dianggap 'Apa itu indonesia?' (memori masih kosong)
    assert "Memori masih kosong" in out and "terlalu pendek" not in out


def test_explicit_commands_still_work(chat, agent):
    assert chat.handle("ingat: Bu Rina tinggal di Jalan Mawar nomor 5.", agent).startswith("Dicatat [1]")
    assert "Bu Rina" in chat.handle("daftar", agent)
    assert "dilupakan" in chat.handle("lupakan 1", agent)
    assert "Tidak ada catatan aktif" in chat.handle("lupakan 1", agent)
    assert chat.handle("keluar", agent) == "__exit__"
    assert chat.handle("   ", agent) is None


def test_is_question_like():
    mod = load_module("memory_chat_under_test2", "tools/memory_chat.py")
    assert mod.is_question_like("Siapa istri Pak Joko")
    assert mod.is_question_like("di mana rumah Bu Rina")
    assert mod.is_question_like("Berapa harga donat")
    assert not mod.is_question_like("Bu Rina tinggal di Jalan Mawar nomor 5")
    assert not mod.is_question_like("halo")
