"""Paket kecil uji kelulusan untuk Kaggle: harus mandiri (tanpa repo penuh, tanpa torch/config/model) dan tidak kekurangan berkas."""
import subprocess
import sys
import zipfile

from conftest import load_module

KIT = load_module("package_acceptance_kit_under_test", "tools/package_acceptance_kit.py")


def test_kit_runs_standalone_with_a_fake_backend(tmp_path):
    z = KIT.build(tmp_path / "kit.zip")
    root = tmp_path / "kit"
    zipfile.ZipFile(z).extractall(root)
    names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert {"tools/model_acceptance_eval.py", "tools/kamus_bawaan.txt", "evalsets/acceptance_criteria.json", "PETUNJUK.txt"} <= names
    assert not any("checkpoint" in n or n.endswith(".pt") for n in names)
    script = tmp_path / "jalan.py"
    script.write_text(
        "import sys, json\n"
        "sys.modules['torch'] = None  # paket tidak boleh bergantung pada torch\n"
        f"sys.path.insert(0, {str(root / 'tools')!r})\n"
        "import model_acceptance_eval as EV\n"
        "class Tiruan(EV.Backend):\n"
        "    name = 'tiruan'\n"
        "    def choose_tool(self, m): return 'cari'\n"
        "    def read(self, ctx, q): return EV.REFUSAL\n"
        "res = EV.run_all(Tiruan(), only=['router', 'qa_beku', 'kasus', 'ketahanan'], quick=True)\n"
        "assert res['kriteria'] and 'router' in res and 'kasus' in res\n"
        "print('OK', len(res['kriteria']), 'kriteria')\n",
        encoding="utf-8")
    r = subprocess.run([sys.executable, "-I", str(script)], cwd=str(root), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-800:]


def test_kit_lists_every_evalset_the_script_reads():
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "tools" / "model_acceptance_eval.py").read_text(encoding="utf-8")
    used = set(re.findall(r'"(memory_[a-z0-9_]+\.jsonl?)"', src)) | {"acceptance_criteria.json"}
    assert used <= set(KIT.EVALSETS), used - set(KIT.EVALSETS)


def test_acceptance_notebook_is_valid_and_its_code_cells_compile():
    import ast
    import json
    from pathlib import Path
    nb = json.loads((Path(__file__).resolve().parent.parent / "kaggle" / "hai07_model_acceptance.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) >= 5
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, c in enumerate(code):
        ast.parse("".join(c["source"]), filename=f"sel-{i}")                 # sintaks valid
    joined = "\n".join("".join(c["source"]) for c in code)
    assert "model_acceptance_eval.py" in joined and "--backend" in joined and "hai07-acceptance" in joined
    assert "QUICK = True" in "".join(code[0]["source"])                       # uji asap lebih dulu secara bawaan


def test_notebook_summary_logic_works_on_a_real_report():
    """Sel ringkasan notebook dijalankan terhadap laporan JSON nyata dari repo (tanpa Kaggle)."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    nb = json.loads((root / "kaggle" / "hai07_model_acceptance.ipynb").read_text(encoding="utf-8"))
    summary = "".join([c for c in nb["cells"] if c["cell_type"] == "code"][-1]["source"])
    report = root / "reports" / "acceptance_gpt100m.json"
    ns = {"QUICK": False, "OUT": str(root / "reports"), "results": {"x/model-uji": {"kode": 0, "detik": 1, "json": str(report), "galat": ""},
                                                                    "y/model-galat": {"kode": 1, "detik": 1, "json": None, "galat": "Traceback\nOSError: gagal unduh"}}}
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(summary, ns)
    out = buf.getvalue()
    assert "model-uji" in out and "8 dari 14" in out and "GALAT: OSError: gagal unduh" in out and "UJI PENUH" in out
