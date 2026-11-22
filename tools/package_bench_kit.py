"""
Bungkus paket kecil untuk mengukur efisiensi pelatihan di Kaggle/GPU sewaan: apps/*.py (model), tools/bench_train.py, kosakata prod_v2.
  python tools/package_bench_kit.py        # menulis kaggle/hai07-bench.zip (ikut .gitignore; unggah sebagai Kaggle Dataset 'hai07-bench')
Notebook pemakainya: kaggle/hai07_bench_train.ipynb
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB = ["vocab_prod.json", "token2id_prod.json", "id2token_prod.json"]


def build(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((PROJECT_ROOT / "apps").glob("*.py")):
            z.write(f, f"apps/{f.name}")
        for f in sorted((PROJECT_ROOT / "apps" / "modules").glob("*.py")):
            z.write(f, f"apps/modules/{f.name}")
        z.write(PROJECT_ROOT / "tools" / "bench_train.py", "tools/bench_train.py")
        for f in VOCAB:
            z.write(PROJECT_ROOT / "data" / "prod_v2" / f, f"data/prod_v2/{f}")
        z.writestr("PETUNJUK.txt", "Ekstrak, lalu dari folder itu: python tools/bench_train.py --preset 100m --device cuda --amp\nNotebook: kaggle/hai07_bench_train.ipynb\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "kaggle" / "hai07-bench.zip"))
    args = ap.parse_args()
    out = build(args.out)
    print(f"[kit] {out}  ({out.stat().st_size / 1e6:.2f} MB)  sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
