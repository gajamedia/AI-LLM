"""
Bungkus paket kecil untuk menjalankan uji kelulusan model di Kaggle atau GPU sewaan (tanpa repo penuh, tanpa bobot model sekarang).
Isi: 5 berkas kode (tools/), kosakata bawaan, seluruh set uji dan kriteria (evalsets/), dan PETUNJUK.txt.
  python tools/package_acceptance_kit.py            # menulis kaggle/hai07-acceptance.zip (ikut .gitignore; unggah sebagai Kaggle Dataset)
Notebook pemakainya: kaggle/hai07_model_acceptance.ipynb
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE = ["model_acceptance_eval.py", "memory_agent.py", "memory_tools.py", "rag_lib.py", "kamus_bawaan.txt"]
EVALSETS = ["memory_handwritten.jsonl", "memory_handwritten_unans.json", "memory_complex.jsonl", "memory_complex_unans.jsonl",
            "memory_cases.jsonl", "memory_router.jsonl", "memory_router_heldout.jsonl", "memory_router_test2.jsonl", "acceptance_criteria.json"]
PETUNJUK = """Paket uji kelulusan model (hai07 Memori)
=========================================
Isi: tools/ (5 berkas), evalsets/ (set uji + kriteria). Ekstrak ke folder kerja, lalu dari folder itu:

  pip install transformers accelerate
  python tools/model_acceptance_eval.py --backend hf --model <id atau folder model> --quick      # uji asap dulu
  python tools/model_acceptance_eval.py --backend hf --model <id atau folder model>              # uji penuh

Hasil: tabel kriteria LULUS/GAGAL di layar dan laporan JSON di reports/. Kriteria: evalsets/acceptance_criteria.json.
Pembanding model sekarang (gpt_sft_best.pt): 8 dari 14 kriteria gagal; pemilih alat ~17%, nama tim 0 dari 3.
Backend 'gpt100m' tidak termasuk di paket ini (butuh bobot dan kode model sendiri).
Periksa LISENSI setiap model sebelum dipakai untuk produk. Notebook: kaggle/hai07_model_acceptance.ipynb di repo.
"""


def build(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE:
            z.write(PROJECT_ROOT / "tools" / f, f"tools/{f}")
        for f in EVALSETS:
            z.write(PROJECT_ROOT / "evalsets" / f, f"evalsets/{f}")
        z.writestr("PETUNJUK.txt", PETUNJUK)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "kaggle" / "hai07-acceptance.zip"))
    args = ap.parse_args()
    out = build(args.out)
    print(f"[kit] {out}  ({out.stat().st_size / 1e3:.0f} KB)  sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
