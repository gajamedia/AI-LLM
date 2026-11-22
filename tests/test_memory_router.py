"""Pemilih alat buatan sendiri (tools/memory_router.py dan tools/train_router.py)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import load_module

ROOT = Path(__file__).resolve().parent.parent
MT = load_module("memory_tools_under_test_r", "tools/memory_tools.py")
MR = load_module("memory_router_under_test", "tools/memory_router.py")
TR = load_module("train_router_under_test", "tools/train_router.py")
ROUTER = MR.Router.load()


def test_normalize_maps_dates_numbers_number_words_and_months():
    assert MR.normalize("Hari apa 17 Agustus 2026?") == ["hari", "apa", "<date>", "?"]
    assert MR.normalize("berapa dua ratus ditambah 3,5") == ["berapa", "<nw>", "<nw>", "ditambah", "<n>"]
    assert MR.normalize("urutkan 5, 3 dan 9") == ["urutkan", "<n>", "<n>", "dan", "<n>"]
    assert MR.normalize("hitung 7 x 8") == ["hitung", "<n>", "*", "<n>"]
    assert MR.normalize("sampai jumpa di Maret nanti") == ["sampai", "jumpa", "di", "<month>", "nanti"]


def test_features_are_deterministic_and_l2_normalised():
    i1, v1 = MR.features("urutkan angka 5 3 9")
    i2, v2 = MR.features("urutkan angka 5 3 9")
    assert (i1 == i2).all() and np.isclose((v1 ** 2).sum(), 1.0)
    out = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, 'tools'); import memory_router as M; print(M.features('urutkan angka 5 3 9')[0].sum())"],
                         cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and int(out.stdout.strip()) == int(i1.sum())          # hash stabil antar proses (bukan hash() Python yang diacak)


@pytest.mark.parametrize("msg,label", [
    ("urutkan 5, 3, 9, 1", "urut"), ("berapa 12 ditambah 30", "hitung"), ("hari apa 17 Agustus 2026", "tanggal"),
    ("Di mana Bu Rina tinggal?", "cari"), ("halo kak", "sapaan"), ("tuliskan puisi tentang hujan", "tolak"),
    ("Siapa yang menyusun jadwal rapat?", "cari"), ("Berapa total biaya rapat tim?", "cari"), ("Hari apa Bu Rina libur?", "cari"),
])
def test_known_messages_are_routed_correctly(msg, label):
    pred, p = ROUTER.decide(msg, ROUTER.meta["tau"], ROUTER.meta["tau_sapaan"], ROUTER.meta["tau_tolak"])
    assert pred == label, (msg, dict(zip(MR.LABELS, np.round(p, 2))))


def test_decide_falls_back_to_search_when_not_confident():
    W = np.zeros((MR.N_BUCKETS, 6), dtype=np.float32)
    flat = MR.Router(W, np.array([0.1, 0, 0, 0, 0, 0], dtype=np.float32))              # semua kelas hampir seimbang
    label, p = flat.decide("urutkan 3 1 2", tau=0.9)
    assert label == "cari" and p.max() < 0.9
    assert flat.decide("urutkan 3 1 2", tau=0.1)[0] == "urut"


def test_weights_file_is_small_and_loads_without_pickle():
    f = MR.WEIGHTS_FILE
    assert f.exists() and f.stat().st_size < 10_000_000
    z = np.load(f, allow_pickle=False)
    assert z["W"].shape == (MR.N_BUCKETS, 6) and z["b"].shape == (6,)
    assert {"tau", "tau_sapaan", "tau_tolak"} <= set(json.loads(str(z["meta"])))


def test_policy_from_probabilities_and_wilson_interval():
    P = np.array([[0.9, 0.02, 0.02, 0.02, 0.02, 0.02], [0.35, 0.3, 0.1, 0.1, 0.1, 0.05], [0.05, 0.05, 0.05, 0.8, 0.03, 0.02], [0.02, 0.02, 0.02, 0.02, 0.5, 0.42]])
    assert TR.policy_from_P(P, tau=0.4, tau_g=0.6, tau_r=0.5) == ["urut", "cari", "cari", "cari"]
    lo, hi = TR.wilson(190, 200)
    assert 90 < lo < 95 < hi < 98 and TR.wilson(0, 0) == (0.0, 0.0)


def test_classifier_meets_the_development_gates_on_the_frozen_sets():
    """Pintu pengembangan (bukan bukti akhir): memakai set beku yang tracked di git."""
    def load(n):
        return [json.loads(l) for l in (ROOT / "evalsets" / n).read_text(encoding="utf-8").splitlines() if l.strip()]
    tau = (ROUTER.meta["tau"], ROUTER.meta["tau_sapaan"], ROUTER.meta["tau_tolak"])
    accept = [r for n in ("memory_router.jsonl", "memory_router_heldout.jsonl", "memory_router_test2.jsonl") for r in load(n)]
    jeb = load("router_jebakan.jsonl")
    pa = [ROUTER.decide(r["message"], *tau)[0] for r in accept]
    pj = [ROUTER.decide(r["message"], *tau)[0] for r in jeb]
    assert sum(p == r["label"] for p, r in zip(pa, accept)) / len(accept) >= 0.93
    assert not [r for p, r in zip(pa, accept) if r["label"] == "cari" and p != "cari"]
    assert sum(p != "cari" for p in pj) / len(jeb) <= 0.01


# ---------- integrasi ke aplikasi: classifier mengusulkan, alat memverifikasi ----------
def test_hinted_tools_read_loose_phrasings_correctly():
    T = MT
    assert T.route("mau tahu 21 Okt 2023 jatuh di hari apa", None, None) is None                       # aturan ketat tidak membacanya
    assert T.route("mau tahu 21 Okt 2023 jatuh di hari apa", None, None, hint="tanggal")["answer"] == "Sabtu"
    assert T.route("beda 12 Jun 2027 dengan 27 Agustus 2023 berapa hari ya kak", None, None, hint="tanggal")["answer"] == "1385 hari"
    assert T.route("85 lalu dikali 43,7, berapa jadinya", None, None, hint="hitung")["answer"] == "3714,5"
    assert T.route("berapa jumlahnya kalau 571 digabung 98549", None, None, hint="hitung")["answer"] == "99120"
    assert T.route("naikkan urutannya deretan ini: 13,3, 36, tujuh puluh enam, 816", None, None, hint="urut")["answer"] == "13,3; 36; 76; 816"


def test_hint_never_invents_an_answer_when_the_tool_cannot_verify():
    for q, h in (("Siapa yang membuat laporan bulanan?", "hitung"), ("kapan rapat besok", "tanggal"), ("rapikan nomor rumah tadi", "urut"), ("berapa ya", "hitung")):
        assert MT.route(q, None, None, hint=h) is None, q
    assert MT.route("Berapa jumlah anggota koperasi?", None, None, hint="hitung") is None             # pertanyaan catatan tidak boleh jadi hitungan


def test_hint_does_not_change_what_the_strict_rules_already_answer():
    for q in ("urutkan 5, 3, 9, 1", "hitung 12 x 5", "hari apa 1 Mei 2026"):
        assert MT.route(q, None, None) == MT.route(q, None, None, hint="urut") == MT.route(q, None, None, hint="hitung")


def test_assisted_route_modes(monkeypatch, tmp_path):
    MR._ROUTER = None
    q = "bantu hitungkan 70220 ditambah 272"
    assert MR.assisted_route(q, None, None, mode="off") is None                                        # perilaku lama
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv(MR.SHADOW_LOG_ENV, str(log))
    assert MR.assisted_route(q, None, None, mode="shadow") is None                                     # bayangan: tidak mengubah hasil
    rec = log.read_text(encoding="utf-8")
    assert '"classifier": "hitung"' in rec and "70220" not in rec and "bantu" not in rec               # tidak menyimpan teks pesan
    assert MR.assisted_route(q, None, None, mode="on")["answer"] == "70492"
    assert MR.assisted_route("Di mana Bu Rina tinggal?", None, None, mode="on") is None                # pertanyaan catatan tetap ke pencarian
    assert MR.assisted_route("tolong tuliskan puisi tentang laut", None, None, mode="on")["kind"] == "task"


def test_default_mode_is_off_and_missing_weights_are_safe(monkeypatch):
    monkeypatch.delenv(MR.MODE_ENV, raising=False)
    assert MR.get_mode() == "off" and MR.get_mode("sembarang") == "off"
    monkeypatch.setattr(MR, "_ROUTER", False)                                                          # bobot hilang: pakai aturan lama
    assert MR.assisted_route("bantu hitungkan 1 ditambah 2", None, None, mode="on") is None
    assert MR.assisted_greeting("selamt pagi", mode="on") is False


def test_assisted_greeting_catches_typos_only_for_short_greetings():
    MR._ROUTER = None
    assert MR.assisted_greeting("selamt pagi", mode="on") is True
    assert MR.assisted_greeting("selamt pagi", mode="off") is False
    assert MR.assisted_greeting("Di mana Bu Rina tinggal?", mode="on") is False
    assert MR.assisted_greeting("haloo 2 3", mode="on") is False


# ---------- jawaban terpotong di tengah kata ----------
MA = load_module("memory_agent_under_test_snap", "tools/memory_agent.py")


def test_snap_repairs_words_cut_at_either_edge_and_keeps_whole_answers():
    note = "Festival layang-layang tahun ini diadakan di pantai Parangtritis pada bulan Agustus."
    assert MA.snap_to_word_boundaries("pantai Parangtriti", note) == "pantai Parangtritis"
    assert MA.snap_to_word_boundaries("angtritis", note) == "Parangtritis"
    assert MA.snap_to_word_boundaries("parangtriti.", note) == "Parangtritis"
    assert MA.snap_to_word_boundaries("Parangtritis", note) == "Parangtritis"        # sudah utuh
    assert MA.snap_to_word_boundaries("layang", note) == "layang"                     # utuh di tepi kata (tanda hubung)


def test_snap_never_guesses_when_it_is_not_a_clear_cut_word():
    note = "Rapat diadakan hari Selasa. Dokumen diadakan lagi nanti."
    assert MA.snap_to_word_boundaries("ada", note) == "ada"                           # terlalu pendek
    assert MA.snap_to_word_boundaries("Dokum", note) == "Dokumen"
    assert MA.snap_to_word_boundaries("tidak ada di catatan", note) == "tidak ada di catatan"   # bukan potongan catatan
    assert MA.snap_to_word_boundaries("dia", "Surat untuk si Dia tiba hari Rabu.") == "dia"      # utuh (huruf besar-kecil diabaikan)
    assert MA.snap_to_word_boundaries("kumen diadakan lagi nanti pada hari", note + " pada hari") != ""
    assert MA.snap_to_word_boundaries("abcde", "xx abcdefghijkl yy") == "abcde"      # perpanjangan > 4 huruf: tidak ditebak
    assert MA.snap_to_word_boundaries("", note) == "" and MA.snap_to_word_boundaries(None, note) is None
