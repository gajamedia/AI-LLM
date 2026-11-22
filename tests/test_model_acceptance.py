"""Alat ukur kelulusan model (tools/model_acceptance_eval.py): diuji dengan model tiruan, tanpa memuat model sungguhan."""
import json
from pathlib import Path

import pytest

from conftest import load_module

EV = load_module("model_acceptance_under_test", "tools/model_acceptance_eval.py")
EVALSETS = Path(__file__).resolve().parent.parent / "evalsets"


def _rows(name):
    return [json.loads(l) for l in open(EVALSETS / name, encoding="utf-8") if l.strip()]


GOLD_TOOL = {r["message"]: r["label"] for f in EV.ROUTER_SETS.values() for r in _rows(f)}


class OracleBackend(EV.Backend):
    """Model tiruan yang sempurna: pemilih alat memakai label emas; pembaca mengutip jawaban emas KASUS YANG SEDANG DIJALANKAN bila ada di
    catatan, selain itu menolak (agen menulis ulang pertanyaan, jadi teks pertanyaan tidak bisa dipakai sebagai kunci)."""
    name = "oracle"
    current = None

    def choose_tool(self, message):
        return GOLD_TOOL[message]

    def read(self, ctx, question):
        c = self.current or {}
        if c.get("clarify"):
            return "Anggrek" if "Anggrek" in ctx else "Mawar" if "Mawar" in ctx else EV.REFUSAL
        gold = c.get("answer")
        return gold if gold and gold.lower() in ctx.lower() else EV.REFUSAL


class AlwaysUrutBackend(EV.Backend):
    """Meniru perilaku model sekarang: selalu memilih kelas pertama dan selalu menolak membaca."""
    name = "selalu-urut"

    def choose_tool(self, message):
        return "urut"

    def read(self, ctx, question):
        return EV.REFUSAL


def test_router_sets_are_balanced_enough_to_expose_a_constant_predictor():
    res = EV.run_router(AlwaysUrutBackend())
    for name, r in res.items():
        assert r["akurasi"] < 0.25, (name, r["akurasi"])                # satu kelas tetap tidak boleh tampak bagus
        assert r["pembajakan_cari"] == sum(1 for x in _rows(EV.ROUTER_SETS[name]) if x["label"] == "cari")


def test_quick_router_sample_covers_all_classes_not_just_the_first_one():
    seen = {}

    class Spy(EV.Backend):
        def choose_tool(self, message):
            seen[GOLD_TOOL[message]] = seen.get(GOLD_TOOL[message], 0) + 1
            return "urut"

    EV.run_router(Spy(), limit=6)
    assert set(seen) == set(EV.LABELS)                                   # dulu: 6 pesan pertama semuanya kelas 'urut'


def test_oracle_router_passes_and_constant_router_fails_the_criteria():
    ok = EV.evaluate_criteria({"router": EV.run_router(OracleBackend())}, EV.CRITERIA)
    assert ok and all(k[3] for k in ok)
    bad = EV.evaluate_criteria({"router": EV.run_router(AlwaysUrutBackend())}, EV.CRITERIA)
    assert not any(k[3] for k in bad)


def test_cases_with_oracle_reader_mostly_pass_and_refusing_reader_fails(monkeypatch):
    oracle, orig = OracleBackend(), EV._case_agent

    def with_current(case, reader):
        oracle.current = case
        return orig(case, reader)

    monkeypatch.setattr(EV, "_case_agent", with_current)
    good_rows, good_rob = EV.run_cases(EV._AgentReader(oracle))
    failed = [r["id"] for r in good_rows if not r["lolos"]]
    assert len(good_rows) == 26 and failed == [], failed                    # pembaca sempurna lewat seluruh alur dan penjaga aplikasi
    assert good_rob["varian_n"] > 0 and good_rob["lolos_varian"] >= 0.85
    bad_rows, _ = EV.run_cases(EV._AgentReader(AlwaysUrutBackend()))
    answerable = [r for r in bad_rows if r["jenis"] == "jawab"]
    assert not any(r["lolos"] for r in answerable)                         # model yang selalu menolak tidak lulus satu pun kasus jawab
    assert all(r["lolos"] for r in bad_rows if r["jenis"] == "tolak")      # tetapi lulus pertanyaan yang harus ditolak (diam itu benar)


def test_full_report_has_decision_and_quick_mode_marks_criteria_as_not_applicable(tmp_path):
    res = EV.run_all(AlwaysUrutBackend(), only=["router", "kasus"], quick=True)
    assert res["cepat"] is True and "kriteria" in res and res["semua_lolos"] is False
    out = tmp_path / "r.json"
    code = EV.main(["--backend", "gpt100m", "--help"]) if False else 0
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8"))["model"] == "selalu-urut"


def test_criteria_can_be_overridden_and_empty_refusal_list_does_not_pass_vacuously():
    rows = [{"id": "x", "tag": "segar", "jenis": "jawab", "lolos": True, "status": "ok", "jawaban": "a"}]
    crit = json.loads(json.dumps(EV.CRITERIA))
    got = {k[0]: k[3] for k in EV.evaluate_criteria({"kasus": rows}, crit)}
    assert got["kasus: pertanyaan yang harus ditolak"] is False             # tidak ada kasus tolak: tidak lulus diam-diam
    crit["kasus"]["lolos_min"] = 1
    assert {k[0]: k[3] for k in EV.evaluate_criteria({"kasus": rows}, crit)}["kasus: lolos"] is True


def test_hf_backend_gives_clear_message_without_transformers(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("transformers"):
            raise ImportError("tidak ada")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(SystemExit, match="pip install transformers"):
        EV.HFBackend("model-apa-saja")


def test_label_in_picks_first_label_mentioned():
    assert EV.label_in("Alat yang tepat adalah hitung.") == "hitung"
    assert EV.label_in("cari lalu urut") == "cari"
    assert EV.label_in("Teks tidak menyebutkan hal itu.") is None and EV.label_in(None) is None


def test_approved_criteria_file_matches_the_code_fallback_and_is_used_by_default():
    approved = json.loads((EVALSETS / "acceptance_criteria.json").read_text(encoding="utf-8"))
    assert "disetujui" in approved["_catatan"].lower()
    assert {k: v for k, v in approved.items() if not k.startswith("_")} == EV.CRITERIA   # satu sumber kebenaran, tidak melenceng
    assert EV.load_criteria() == approved
    with pytest.raises(SystemExit):
        EV.load_criteria("tidak-ada.json")
