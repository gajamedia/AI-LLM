import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

for _sub in ("apps", "tools"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)


def load_module(name: str, relpath: str):
    """Load a script under tools/ (they are not a package) as a module."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {relpath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def production_cfg():
    from config import get_config

    return get_config("production")


@pytest.fixture(scope="session")
def production_tokenizer(production_cfg):
    required = (
        Path(str(production_cfg.VOCAB_PATH)),
        Path(str(production_cfg.TOKEN2ID_PATH)),
        Path(str(production_cfg.ID2TOKEN_PATH)),
    )
    if not all(p.exists() for p in required):
        pytest.skip("production tokenizer artifacts not present (data/prod)")

    import tokenizer2

    return tokenizer2.build_tokenizer_from_config(cfg=production_cfg)


@pytest.fixture(scope="session")
def train_module():
    return load_module("train_gptV2_under_test", "tools/train_gptV2.py")


@pytest.fixture(scope="session")
def generate_module():
    return load_module("generate_gpt_under_test", "tools/generate_gpt.py")


@pytest.fixture(autouse=True)
def _tanpa_berkas_konfigurasi(monkeypatch, tmp_path):
    """Tes tidak boleh terpengaruh memori.env milik pengguna."""
    monkeypatch.setenv("MEMORI_CONFIG", str(tmp_path / "tidak_ada.env"))
    try:
        import memory_router
        memory_router.reload_config()
    except ImportError:
        pass
