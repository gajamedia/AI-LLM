"""Pemilih alat buatan sendiri (tools/memory_router.py dan tools/train_router.py)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import load_module

ROOT = Path(__file__).resolve().parent.parent
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
