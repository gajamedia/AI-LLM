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


# ---------- ujian gaya independen: perbaikan setelah melihat hasilnya ----------
def test_no_control_characters_slipped_into_source_files():
    for f in list((ROOT_DIR := Path(__file__).resolve().parent.parent).glob("tools/*.py")) + list((ROOT_DIR / "apps").glob("*.py")):
        raw = f.read_bytes()
        assert not any(c in raw for c in (b"\x08", b"\x07", b"\x0c")), f.name           # '\b' dalam skrip tambal pernah menjadi karakter backspace


def test_loose_calculator_reads_common_word_problems_only_when_hinted():
    T = MT
    cases = {"Berapa total belanja 35 ribu, 18 ribu, dan 42 ribu?": "95000", "Aku punya 250 ribu lalu belanja 87 ribu, sisa uangku berapa?": "163000",
             "Tolong hitung rata-rata dari 70, 80, dan 90": "80", "Diskon 20 persen dari harga 150000 jadi berapa?": "120000",
             "Jumlahkan semua angka ini ya: 5, 10, 15, 20, 25": "75", "Hitung selisih antara 1.250.000 dan 875.000": "375000"}
    for q, ans in cases.items():
        assert T.route(q, None, None, hint="hitung")["answer"] == ans, q
    assert T.route("Siapa yang membuat laporan bulanan 3 kali?", None, None, hint="hitung") is None


def test_assisted_route_policy_on_unreadable_tool_requests_refusals_and_notes(monkeypatch):
    MR._ROUTER = None
    r = MR.assisted_route("Kalau mulai 20 November dan selesai 5 Desember, berapa hari jaraknya?", None, None, mode="on")
    assert r is None or r.get("hint") == "tanggal"                                    # tidak jadi pencarian catatan yang menyesatkan bila classifier yakin
    r = MR.assisted_route("Tolong hitung rata-rata dari 70, 80, dan 90", None, None, mode="on")
    assert r["kind"] == "tool" and r["answer"] == "80"                                # aturan lama menolaknya sebagai 'tugas': kini dihitung
    assert MR.assisted_route("Tolong cari catatan tentang susun jadwal piket minggu depan", None, None, mode="on") is None   # menyebut catatan: bukan permintaan tugas
    assert MR.assisted_route("Tolong buatkan puisi tentang ibu", None, None, mode="on")["kind"] == "task"
    assert MR.assisted_greeting("Hai, apa kabar?", mode="on") is True
    assert MR.assisted_greeting("Di mana Bu Rina tinggal?", mode="on") is False


def test_shadow_log_stores_text_only_when_explicitly_enabled(monkeypatch, tmp_path):
    MR._ROUTER = None
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv(MR.SHADOW_LOG_ENV, str(log))
    monkeypatch.delenv(MR.SHADOW_TEXT_ENV, raising=False)
    MR.assisted_route("bantu hitungkan 70220 ditambah 272", None, None, mode="shadow")
    assert "teks" not in json.loads(log.read_text(encoding="utf-8").splitlines()[0])             # bawaan: tanpa teks
    monkeypatch.setenv(MR.SHADOW_TEXT_ENV, "1")
    out = MR.assisted_route("bantu hitungkan 70220 ditambah 272", None, None, mode="shadow")
    assert out is None                                                                           # bayangan tidak mengubah hasil
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["teks"] == "bantu hitungkan 70220 ditambah 272"


def test_log_tools_roundtrip_without_leaking_the_classifier_guess(tmp_path):
    LT = load_module("router_log_tools_under_test", "tools/router_log_tools.py")
    log = tmp_path / "log.jsonl"
    entries = [{"teks": "tolong urutin 3 1 2", "classifier": "urut"}, {"teks": "Tolong urutin 3 1 2", "classifier": "urut"},
               {"teks": "hai | min", "classifier": "sapaan"}, {"classifier": "cari"}]
    log.write_text("\n".join(json.dumps(d) for d in entries), encoding="utf-8")
    md = tmp_path / "l.md"
    assert LT.to_md(log, md) == 2                                                                # duplikat (beda huruf besar) dan baris tanpa teks dibuang
    table = md.read_text(encoding="utf-8").split("|---|---|---|")[1]
    assert "urut" not in table.replace("urutin", "") and "sapaan" not in table                  # tidak ada tebakan classifier di lembar
    filled = md.read_text(encoding="utf-8").replace("| tolong urutin 3 1 2 | |", "| tolong urutin 3 1 2 | urut |").replace("| hai / min | |", "| hai / min | Sapaan |")
    md.write_text(filled, encoding="utf-8")
    rows, bad = LT.from_md(md, tmp_path / "o.jsonl")
    assert rows == [{"message": "tolong urutin 3 1 2", "label": "urut"}, {"message": "hai / min", "label": "sapaan"}] and not bad


def _feeder(lines):
    it = iter(lines)

    def f(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return f


def test_cli_tulis_saves_messages_with_their_class_and_resumes_without_duplicates(tmp_path):
    LT = load_module("router_log_tools_under_test_tulis", "tools/router_log_tools.py")
    out = tmp_path / "set.jsonl"
    LT.cmd_tulis(out, input_fn=_feeder(["2", "hitung 3 + 4", "tambahin 5 sama 6", "", "5", "hai min", "selesai"]), print_fn=lambda *a: None)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"message": "hitung 3 + 4", "label": "hitung"}, {"message": "tambahin 5 sama 6", "label": "hitung"}, {"message": "hai min", "label": "sapaan"}]
    LT.cmd_tulis(out, input_fn=_feeder(["sapaan", "Hai Min", "/tolak", "bikinin pantun", "/9", "q"]), print_fn=lambda *a: None)   # 'Hai Min' duplikat; /nama ganti kelas; /9 tidak sah
    assert [r["message"] for r in LT._load_set(out)] == ["hitung 3 + 4", "tambahin 5 sama 6", "hai min", "bikinin pantun"]


def test_cli_label_goes_through_the_log_and_skips_discards_and_quits(tmp_path):
    LT = load_module("router_log_tools_under_test_label", "tools/router_log_tools.py")
    log, out = tmp_path / "log.jsonl", tmp_path / "set.jsonl"
    log.write_text("\n".join(json.dumps({"teks": t}) for t in ["urutin 3 1 2", "pesan pribadi", "siapa ketua rt", "jam berapa sekarang"]), encoding="utf-8")
    LT.cmd_label(log, out, input_fn=_feeder(["1", "h", "cari", "q"]), print_fn=lambda *a: None)
    assert LT._load_set(out) == [{"message": "urutin 3 1 2", "label": "urut"}, {"message": "siapa ketua rt", "label": "cari"}]
    LT.cmd_label(log, out, input_fn=_feeder(["s"]), print_fn=lambda *a: None)           # sisa hanya 'pesan pribadi' dan 'jam berapa': tidak dilabeli ulang yang sudah ada
    assert len(LT._load_set(out)) == 2


def test_cli_review_records_answers_saves_progress_and_resumes(tmp_path):
    LT = load_module("router_log_tools_under_test_review", "tools/router_log_tools.py")
    md = tmp_path / "u.md"
    md.write_text("| No | Pesan | Label | Benar? |\n|---|---|---|---|\n| 1 | urutin 3 1 2 | urut | |\n| 2 | hai min | sapaan | |\n| 3 | bikin pantun | tolak | |\n", encoding="utf-8")
    out = tmp_path / "hasil.md"
    done = LT.cmd_review(md, out, input_fn=_feeder(["y", "x", "r", "q"]), print_fn=lambda *a: None)       # 'x' tidak sah: ditanya ulang
    assert done == {"1": "ya", "2": "ragu"}
    done = LT.cmd_review(md, out, input_fn=_feeder(["t"]), print_fn=lambda *a: None)                       # lanjut dari nomor 3
    assert done == {"1": "ya", "2": "ragu", "3": "tidak"}
    assert "ya 1, tidak 1, ragu 1" in out.read_text(encoding="utf-8")


def test_independent_style_set_exists_and_is_wellformed_if_present():
    path = Path(__file__).resolve().parent.parent / "evalsets" / "router_tulisan_anda.jsonl"
    if not path.exists():
        pytest.skip("berkas pesan independen belum ada")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert len(rows) >= 100 and {r["label"] for r in rows} == set(MR.LABELS) and all(r["text"].strip() for r in rows)


def test_all_router_sets_share_the_message_label_format_and_old_text_key_is_still_read(tmp_path):
    LT = load_module("router_log_tools_under_test_fmt", "tools/router_log_tools.py")
    old = tmp_path / "lama.jsonl"
    old.write_text(json.dumps({"text": "hai min", "label": "sapaan"}) + "\n", encoding="utf-8")
    assert LT._load_set(old) == [{"message": "hai min", "label": "sapaan"}]                       # berkas lama dengan kunci 'text' tetap terbaca
    EV = load_module("eval_router_independent_under_test", "tools/eval_router_independent.py")
    out = tmp_path / "r.json"
    EV.main(["--file", str(old), "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["n"] == 1
    for f in (Path(__file__).resolve().parent.parent / "evalsets" / "router_pakai_nyata.jsonl",):
        if f.exists():
            assert all("message" in json.loads(l) and "text" not in json.loads(l) for l in open(f, encoding="utf-8") if l.strip())


def test_penjaga_sapaan_menolak_pesan_tak_berkaitan():
    import memory_router as MR
    for q in ["pandora", "bangunkan saya pukul tujuh pagi", "rasanya harus", "cari film di dekat saya", "batas waktu", "hapus alarm saya"]:
        assert not MR.has_greeting_cue(q), q
        assert not MR.assisted_greeting(q, mode="on"), q


def test_penjaga_sapaan_menerima_varian_salah_ketik():
    import memory_router as MR
    for q in ["saaaalam semua", "dadaaaah semua", "selamt pagi", "thankks a lot kak", "permiiiisi semua", "hlo sob"]:
        assert MR.has_greeting_cue(q), q


def test_berkas_konfigurasi_dan_prioritas(tmp_path, monkeypatch):
    import memory_router as MR
    cfg = tmp_path / "memori.env"
    cfg.write_text('# komentar\nMEMORI_ROUTER=shadow\nMEMORI_ROUTER_LOG_TEXT="1"\nMEMORI_ROUTER_HINT_MIN_P=0.8 # catatan\nrusak tanpa sama dengan\n', encoding="utf-8")
    monkeypatch.setenv("MEMORI_CONFIG", str(cfg))
    monkeypatch.delenv("MEMORI_ROUTER", raising=False)
    MR.reload_config()
    assert MR.get_mode() == "shadow"
    assert MR.setting("MEMORI_ROUTER_LOG_TEXT") == "1"
    assert MR._threshold("MEMORI_ROUTER_HINT_MIN_P", 0.6) == 0.8
    monkeypatch.setenv("MEMORI_ROUTER", "on")                  # variabel lingkungan mengalahkan berkas
    assert MR.get_mode() == "on"
    monkeypatch.delenv("MEMORI_ROUTER")
    cfg.write_text("MEMORI_ROUTER=ngawur\nMEMORI_ROUTER_HINT_MIN_P=abc\n", encoding="utf-8")
    MR.reload_config()
    assert MR.get_mode() == "off"                              # nilai tidak dikenal -> aman (mati)
    assert MR._threshold("MEMORI_ROUTER_HINT_MIN_P", 0.6) == 0.6
    monkeypatch.setenv("MEMORI_CONFIG", str(tmp_path / "tidak_ada.env"))
    MR.reload_config()
    assert MR.get_mode() == "off"                              # tanpa berkas -> bawaan lama
