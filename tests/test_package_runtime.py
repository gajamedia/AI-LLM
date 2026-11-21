"""Penjaga paket rilis: setiap modul lokal yang diimpor oleh kode runtime harus ikut dibundel oleh package_release."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _runtime_tools():
    text = (ROOT / "tools" / "package_release.py").read_text(encoding="utf-8")
    block = re.search(r"RUNTIME_TOOLS\s*=\s*\[(.*?)\]", text, re.S).group(1)
    return re.findall(r'"([\w.]+\.py)"', block)


def _local_imports(path):
    mods = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(?:import|from)\s+(\w+)", line)
        if m:
            mods.add(m.group(1))
    return mods


def test_runtime_tools_cover_all_local_imports():
    bundled = set(_runtime_tools())
    assert "memory_docs.py" in bundled
    missing = {}
    for name in bundled:
        for mod in _local_imports(ROOT / "tools" / name):
            if (ROOT / "tools" / f"{mod}.py").exists() and f"{mod}.py" not in bundled:
                missing.setdefault(name, []).append(mod)
    assert not missing, f"modul tools/ belum dibundel oleh package_release: {missing}"


def test_web_page_is_bundled_and_self_contained():
    text = (ROOT / "tools" / "package_release.py").read_text(encoding="utf-8")
    assert 'web" / "memory" / "index.html"' in text.replace("'", '"')
    html = (ROOT / "web" / "memory" / "index.html").read_text(encoding="utf-8")
    assert not re.search(r'(?:src|href)\s*=\s*"https?://', html), "halaman web tidak boleh memuat sumber daya dari internet"
