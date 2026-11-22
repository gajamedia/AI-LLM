"""SFT tahap 6: pembangkit jenis fakta baru, penjaga kebocoran, dan paket Kaggle."""
import ast
import json
import random
import sys
import zipfile
from pathlib import Path

from conftest import load_module

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import memory_gen3 as G3  # noqa: E402
import rag_lib as R  # noqa: E402


def test_every_gold_answer_is_an_exact_span_of_the_note_and_names_keep_their_case():
    rng = random.Random(1)
    for fam in G3.FAMILIES3:
        for _ in range(200):
            note, qa, un = fam(rng)
            for q, a in qa:
                assert a in note, (fam.__name__, a, note)
        assert un
    assert G3._cap("tim Bajama") == "Tim Bajama"                       # nama di dalam tidak ikut dikecilkan


def test_examples_cover_answerable_and_both_unanswerable_kinds_with_confusable_distractors():
    rng = random.Random(2)
    ex = [G3.make_example3(rng) for _ in range(3000)]
    kinds = {e["kind"] for e in ex}
    assert kinds == {"answerable", "unanswerable_related", "unanswerable_absent"}
    for e in ex:
        if e["answer"] is not None:
            assert e["answer"] in e["ctx"]
    hours = [e for e in ex if "buka" in e["question"] or "tutup" in e["question"]]
    assert len(hours) > 100 and any(e["ctx"].count("pukul") + e["ctx"].count("Jam tutup") >= 2 for e in hours)   # pengecoh sejenis ada


def test_builder_mixes_old_data_and_never_copies_test_questions(tmp_path):
    B = load_module("build_sft3_under_test", "tools/build_sft_memory3_data.py")
    old = tmp_path / "sft_mem2.jsonl"
    rows = [{"prompt": "p", "response": "r", "source": s, "question": "q"} for s in ["mem2_answerable"] * 30 + ["mem_answerable"] * 30 + ["rag_pos"] * 30]
    old.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    train, val = B.main(["--n-v3", "200", "--n-val", "20", "--mem2", str(old), "--keep-mem2", "10", "--keep-mem1", "10", "--keep-replay", "10", "--out-dir", str(tmp_path)])
    src = [r["source"].split("_")[0] for r in train]
    assert src.count("mem3") == 200 and src.count("mem2") == 10 and src.count("mem") == 10 and src.count("rag") == 10
    test_q = set()
    for f in list((ROOT / "evalsets").glob("*.jsonl")) + list((ROOT / "data" / "sft").glob("memory_complex*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                test_q.update(d[k].strip().lower() for k in ("question", "message") if k in d)
    assert not any(r["question"].strip().lower() in test_q for r in train if r["source"].startswith("mem3_"))
    assert (tmp_path / "sft_mem3.jsonl").exists() and (tmp_path / "sft_mem3_val.jsonl").exists()
    assert R.NO_ANSWER and all(r["response"] in R.NO_ANSWER for r in train if r["source"].startswith("mem3_unanswerable"))


def test_sft6_kit_contents_and_notebook_are_consistent(tmp_path):
    K = load_module("package_sft6_under_test", "tools/package_sft6_kit.py")
    z = K.build(tmp_path / "k.zip")
    names = set(zipfile.ZipFile(z).namelist())
    for n in ("hai07-sft6/tools/train_sft.py", "hai07-sft6/tools/memory_gen3.py", "hai07-sft6/data/sft/sft_mem3.jsonl", "hai07-sft6/data/prod/token2id_prod.json", "hai07-sft6/apps/gptModel.py"):
        assert n in names, n
    assert not any(n.endswith((".pt", ".bin")) for n in names)
    nb = json.loads((ROOT / "kaggle" / "hai07_sft_mem3_100m.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(l for l in c["source"] if not l.startswith("!")) for c in nb["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    assert "checkpoints_100m_sft_mem2" in code and "sft_mem3.jsonl" in code and "--data" in code and "hai07-sft6" in code
    assert "hai07-sft5" not in code
