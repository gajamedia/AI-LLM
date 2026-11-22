"""Pembangkit data pemilih alat (tools/make_router_data.py): invarian label, kebocoran, kuota salah ketik, pembekuan set uji."""
import collections
import json
import random
import re
from pathlib import Path

import pytest

from conftest import load_module

MR = load_module("make_router_data_under_test", "tools/make_router_data.py")
EV = Path(__file__).resolve().parent.parent / "evalsets"
LABELS = MR.LABELS


def _load(name):
    return [json.loads(l) for l in open(EV / name, encoding="utf-8") if l.strip()]


FROZEN = {"validasi": _load("router_validasi.jsonl"), "jebakan": _load("router_jebakan.jsonl"), "berisik": _load("router_berisik.jsonl")}
ACCEPT = [r for f in ("memory_router.jsonl", "memory_router_heldout.jsonl", "memory_router_test2.jsonl") for r in _load(f)]
SMALL = MR.build(seed=7, n_train=120, n_train_cari=300, frozen=FROZEN)

NUM = re.compile(r"\d|\b(nol|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|belas|puluh|ratus|ribu)\b", re.I)
DATE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(" + "|".join(MR.MONTHS + MR.MONTH_ABBR) + r")\b", re.I)


def test_families_a_and_b_are_disjoint_and_cover_every_class():
    for lab in LABELS:
        a = {n for n, _ in MR.FAMILIES_A[lab]()}
        b = {n for n, _ in MR.FAMILIES_B[lab]()}
        assert a and b and not (a & b), lab
    train_fams = {r["family"] for r in SMALL["train"]}
    valid_fams = {r["family"] for r in FROZEN["validasi"]}
    assert train_fams and valid_fams and not (train_fams & valid_fams)         # validasi memakai keluarga templat yang tidak ada di latih


@pytest.mark.parametrize("name,rows", [("latih", SMALL["train"]), ("validasi", FROZEN["validasi"])])
def test_label_invariants_hold_for_messages_without_typos(name, rows):
    bad = []
    for r in rows:
        if r.get("salah_ketik"):
            continue                                                              # salah ketik bisa merusak pola; sengaja dikecualikan
        m, lab = r["message"], r["label"]
        if lab in ("urut",) and not (len(NUM.findall(m)) >= 2 or "catatan" in m.lower()):
            bad.append((lab, m))
        if lab == "hitung" and len(NUM.findall(m)) < 2:
            bad.append((lab, m))
        if lab == "tanggal" and not DATE.search(m):
            bad.append((lab, m))
        if lab == "sapaan" and (re.search(r"\d", m) or len(m.split()) > 9):
            bad.append((lab, m))
        if lab == "tolak" and re.search(r"\b\d{3,}\b", m):
            bad.append((lab, m))
    assert not bad, bad[:5]


def test_no_exact_or_near_duplicates_between_train_and_the_test_sets():
    protected = [r["message"] for r in ACCEPT] + [r["message"] for r in FROZEN["validasi"] + FROZEN["jebakan"]]
    exact = {MR._norm(m) for m in protected}
    assert not [r for r in SMALL["train"] if MR._norm(r["message"]) in exact]
    assert len({MR._norm(r["message"]) for r in SMALL["train"]}) == len(SMALL["train"])          # tidak ada duplikat di dalam latih
    sh = [MR._shingles(m) for m in protected]
    rng = random.Random(1)
    for r in rng.sample(SMALL["train"], min(150, len(SMALL["train"]))):
        s = MR._shingles(r["message"])
        assert all(len(s & t) / max(1, len(s | t)) < 0.8 for t in sh), r["message"]


def test_typo_quota_keeps_every_class_mostly_clean():
    for lab in LABELS:
        rows = [r for r in SMALL["train"] if r["label"] == lab]
        assert rows and sum(r["salah_ketik"] for r in rows) / len(rows) <= 0.35, lab           # dulu sapaan hampir seluruhnya salah ketik


def test_sapaan_has_real_diversity_not_a_tiny_closed_list():
    full = MR.generate(MR.FAMILIES_A, "sapaan", 600, random.Random(3))
    assert len(full) >= 450 and len({r["family"] for r in full}) >= 6


def test_polite_prefix_not_doubled_and_not_added_to_greetings():
    rng = random.Random(5)
    for _ in range(400):
        assert not re.match(r"^(tolong|mohon|coba|bisa|kak|min) (tolong|mohon|coba|bisa|kak|min)\b", MR.style(rng, "tolong buatkan puisi"))
    for r in MR.generate(MR.FAMILIES_A, "sapaan", 300, random.Random(9)):
        assert not r["message"].lower().startswith(("tolong ", "mohon ", "coba ", "bisa "))


def test_tolak_objects_match_their_verbs():
    for r in MR.generate(MR.FAMILIES_A, "tolak", 400, random.Random(11)):
        m = r["message"].lower()
        if "rapat kemarin" in m or "isi dokumen" in m or "berita hari ini" in m:                 # objek ringkasan hanya dengan kata kerja ringkas/rangkum
            assert re.search(r"ringkas|rangkum", m) or r["salah_ketik"], m


def test_berisik_set_changes_every_message_and_keeps_its_label():
    origin = {r["message"]: r["label"] for r in ACCEPT}
    assert len(FROZEN["berisik"]) == 3 * len(ACCEPT)
    for r in FROZEN["berisik"]:
        assert r["asal"] in origin and origin[r["asal"]] == r["label"]
    changed = sum(r["message"] != r["asal"] for r in FROZEN["berisik"])
    assert changed / len(FROZEN["berisik"]) >= 0.9


def test_frozen_sets_are_loaded_as_is_and_have_the_planned_size():
    again = MR.build(seed=99, n_train=60, n_train_cari=100, frozen=FROZEN)
    assert again["validasi"] == FROZEN["validasi"] and again["jebakan"] == FROZEN["jebakan"] and again["berisik"] == FROZEN["berisik"]
    assert len(FROZEN["validasi"]) >= 1500 and len(FROZEN["jebakan"]) >= 120
    assert all(r["label"] == "cari" for r in FROZEN["jebakan"])
    assert collections.Counter(r["label"] for r in FROZEN["validasi"]).keys() == set(LABELS)


def test_length_balance_between_non_greeting_classes_is_close():
    means = {lab: sum(len(r["message"].split()) for r in SMALL["train"] if r["label"] == lab) / sum(r["label"] == lab for r in SMALL["train"])
             for lab in LABELS if lab != "sapaan"}
    assert max(means.values()) - min(means.values()) <= 3.0, means                           # tanpa penyeimbangan: urut 7 vs cari 5 ke atas
