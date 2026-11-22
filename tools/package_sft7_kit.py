"""
Paket Kaggle untuk SFT tahap 7 (v7): kode model, alat latih, pembangkit data, data sft_mem4, kosakata.
  python tools/build_sft_memory4_data.py     # membuat data/sft/sft_mem4*.jsonl dulu
  python tools/package_sft6_kit.py           # menulis kaggle/hai07-sft7.zip (ikut .gitignore; unggah sebagai Kaggle Dataset 'hai07-sft7')
Notebook pemakainya: kaggle/hai07_sft_mem4_100m.ipynb (basis: output SFT tahap 5 (BUKAN v6), folder checkpoints_100m_sft_mem2).
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS = ["train_sft.py", "train_gptV2.py", "rag_lib.py", "memory_gen.py", "memory_gen2.py", "memory_gen3.py", "memory_gen4.py", "build_sft_memory4_data.py",
         "memory_agent.py", "memory_tools.py", "memory_router.py", "generate_gpt.py"]
VOCAB = ["vocab_prod.json", "token2id_prod.json", "id2token_prod.json"]
DATA = ["sft_mem4.jsonl", "sft_mem4_val.jsonl"]
ROOT = "hai07-sft7"


def build(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in ("apps", "apps/modules"):
            for f in sorted((PROJECT_ROOT / d).glob("*.py")):
                z.write(f, f"{ROOT}/{d}/{f.name}")
        for name in TOOLS:
            z.write(PROJECT_ROOT / "tools" / name, f"{ROOT}/tools/{name}")
        z.write(PROJECT_ROOT / "tools" / "router_weights.npz", f"{ROOT}/tools/router_weights.npz")
        z.write(PROJECT_ROOT / "tools" / "kamus_bawaan.txt", f"{ROOT}/tools/kamus_bawaan.txt")
        for f in VOCAB:
            z.write(PROJECT_ROOT / "data" / "prod_v2" / f, f"{ROOT}/data/prod/{f}")
        for f in DATA:
            z.write(PROJECT_ROOT / "data" / "sft" / f, f"{ROOT}/data/sft/{f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "kaggle" / "hai07-sft7.zip"))
    args = ap.parse_args()
    out = build(args.out)
    print(f"[kit] {out}  ({out.stat().st_size / 1e6:.2f} MB)  sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
