"""SFT tahap 7: pembangkit v7, penjaga kebocoran terhadap kasus segar, builder, dan paket."""
import ast
import json
import random
import re
import sys
import zipfile
from pathlib import Path

from conftest import load_module

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import memory_gen4 as G4  # noqa: E402

FRESH = [json.loads(l) for l in open(ROOT / "evalsets" / "memory_cases_v7_segar.jsonl", encoding="utf-8") if l.strip()]


def test_every_gold_answer_is_a_span_of_its_note_in_all_families():
    rng = random.Random(4)
    for fam in G4.ALL_FAMILIES:
        for _ in range(250):
            note, qa, un = fam(rng)
            assert all(a in note for _q, a in qa), fam.__name__
            assert un


def test_reverse_questions_and_full_phrases_are_present():
    rng = random.Random(5)
    seen_rev = seen_phrase = seen_end = 0
    for _ in range(400):
        n, qa, _ = G4.fam_twin_events(rng)
        seen_rev += any(q.startswith("tim apa yang") and a in G4.TEAM_WORDS for q, a in qa)
        n, qa, _ = G4.fam_place_phrase(rng)
        seen_phrase += any(" " in a and a.split()[0] in G4.PLACE_TYPES for q, a in qa)
        n, qa, _ = G4.fam_period(rng)
        seen_end += any(q.startswith("Sampai kapan") for q, a in qa)
    assert seen_rev > 300 and seen_phrase > 150 and seen_end > 300


def test_generator_never_uses_the_entities_of_the_fresh_held_out_cases_nor_the_frozen_nyata_cases():
    vocab = " ".join(sum((G4.TEAM_WORDS, G4.SUBJECTS, G4.JOBS, G4.GOODS, G4.PLACE_TYPES), [])).lower()
    rng = random.Random(6)
    text = " ".join(G4.make_note4(rng)[0] for _ in range(4000)).lower()          # hanya famili v6/v7 (famili tahap 5 sudah ada sebelumnya)
    assert not any(re.search(rf"\b{re.escape(e)}\b", text) for e in {"audit", "marketing", "cempaka", "seruni", "sukamaju", "tanjung mas", "kopi hitam", "roti keju"}), "entitas kasus segar bocor"
    assert "finance" not in text and "tim produk" not in text      # kasus nyata rapat-tim-*: pola dilatih, nama tim tidak (nama ruang sudah ada di data tahap 5)
    assert vocab


def test_fresh_cases_are_wellformed_and_answers_are_in_their_notes():
    ids = [c["id"] for c in FRESH]
    assert len(ids) == len(set(ids)) >= 14
    for c in FRESH:
        assert c["tag"] == "segar_v7" and c["answer"].lower() in " ".join(c["notes"]).lower()


def test_builder4_mixes_more_old_data_and_leaks_no_fresh_question(tmp_path):
    B = load_module("build_sft4_under_test", "tools/build_sft_memory4_data.py")
    old = tmp_path / "sft_mem2.jsonl"
    rows = [{"prompt": "p", "response": "r", "source": s, "question": "q"} for s in ["mem2_answerable"] * 40 + ["mem_answerable"] * 40 + ["rag_pos"] * 40]
    old.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    train, val = B.main(["--n-v4", "300", "--n-val", "20", "--mem2", str(old), "--keep-mem2", "20", "--keep-mem1", "10", "--keep-replay", "10", "--out-dir", str(tmp_path)])
    src = [r["source"].split("_")[0] for r in train]
    assert src.count("mem4") == 300 and src.count("mem2") == 20 and src.count("mem") == 10 and src.count("rag") == 10
    fresh_q = {c["question"].strip().lower() for c in FRESH}
    assert not any(r["question"].strip().lower() in fresh_q for r in train if r["source"].startswith("mem4_"))


def test_sft7_kit_and_notebook_are_consistent(tmp_path):
    K = load_module("package_sft7_under_test", "tools/package_sft7_kit.py")
    names = set(zipfile.ZipFile(K.build(tmp_path / "k.zip")).namelist())
    for n in ("hai07-sft7/tools/memory_gen4.py", "hai07-sft7/tools/memory_gen3.py", "hai07-sft7/data/sft/sft_mem4.jsonl", "hai07-sft7/tools/train_sft.py"):
        assert n in names, n
    nb = json.loads((ROOT / "kaggle" / "hai07_sft_mem4_100m.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(l for l in c["source"] if not l.startswith("!")) for c in nb["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    assert 'LR = "1e-5"' in code and "checkpoints_100m_sft_mem4" in code and "sft_mem4.jsonl" in code and "hai07-sft7" in code
    assert "checkpoints_100m_sft_mem2" in code and "hai07-sft6" not in code and "sft_mem3" not in code      # basis tetap tahap 5, bukan v6
