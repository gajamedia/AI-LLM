"""
Bungkus kode prapelatihan 540M untuk Kaggle: apps/*.py, tools/train_gptV2.py, dan kosakata prod_v2 (ditaruh di data/prod/ sesuai konfigurasi).
  python tools/package_pretrain_kit.py     # menulis kaggle/hai07-pretrain-kit.zip (ikut .gitignore; unggah sebagai Kaggle Dataset 'hai07-pretrain-kit')
Data latih (shard korpus skala) TIDAK ikut paket ini: unggah terpisah (lihat kaggle/hai07_pretrain_540m_t4x2.ipynb).
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
        z.write(PROJECT_ROOT / "tools" / "train_gptV2.py", "tools/train_gptV2.py")
        for f in VOCAB:
            z.write(PROJECT_ROOT / "data" / "prod_v2" / f, f"data/prod/{f}")
        z.writestr("PETUNJUK.txt", "Notebook: kaggle/hai07_pretrain_540m_t4x2.ipynb\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "kaggle" / "hai07-pretrain-kit.zip"))
    args = ap.parse_args()
    out = build(args.out)
    print(f"[kit] {out}  ({out.stat().st_size / 1e6:.2f} MB)  sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
