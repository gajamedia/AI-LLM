"""
Membuat paket rilis yang bisa didistribusikan: bobot saja + kode jalan minimal + antarmuka web + dokumen.

  python tools/package_release.py --version 1.1126.9

Keluaran: dist/hai07-memori-v<versi>/ dan dist/hai07-memori-v<versi>.zip (+ SHA256SUMS.txt di dalamnya).
Paket tidak berisi optimizer, data latih, atau catatan pengguna. Berkas lisensi diambil dari licenses/
(kode MIT, bobot CC BY-NC-SA 4.0); bila folder itu tidak ada, paket memuat pernyataan 'belum ditetapkan'.
"""
import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

RUNTIME_TOOLS = ["memory_web.py", "memory_agent.py", "memory_chat.py", "memory_tools.py", "memory_docs.py", "rag_lib.py", "generate_gpt.py"]
VOCAB_FILES = ["vocab_prod.json", "token2id_prod.json", "id2token_prod.json"]
DEFAULT_CKPT_REPO = "reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt"
MODEL_FILE = "model/hai07-100m-v{v}.pt"

REQUIREMENTS = "# Kebutuhan runtime (diuji pada Python 3.10, torch 2.14 CPU)\ntorch>=2.6\nnumpy>=2.0\n"

BAT = r"""@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 atau lebih baru belum terpasang. Unduh dari https://www.python.org/downloads/
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Menyiapkan lingkungan Python sekali saja ^(perlu internet, sekitar 250 MB^)...
  python -m venv .venv || goto :err
  ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :err
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :err
)
".venv\Scripts\python.exe" tools\memory_web.py --open %*
exit /b 0
:err
echo Gagal menyiapkan lingkungan. Periksa koneksi internet lalu jalankan lagi.
pause
exit /b 1
"""

SH = """#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=python3
command -v "$PY" >/dev/null 2>&1 || { echo "Python 3.10+ belum terpasang."; exit 1; }
if [ ! -x ".venv/bin/python" ]; then
  echo "Menyiapkan lingkungan Python sekali saja (perlu internet)..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  if [ "$(uname)" = "Linux" ]; then
    .venv/bin/python -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
    .venv/bin/python -m pip install "numpy>=2.0"
  else
    .venv/bin/python -m pip install -r requirements.txt
  fi
fi
exec .venv/bin/python tools/memory_web.py --open "$@"
"""

NO_LICENSE = """LISENSI BELUM DITETAPKAN

Paket ini dibuat tanpa memilih lisensi. Pemilik harus menetapkan lisensi untuk kode dan untuk bobot model sebelum
membagikannya, dengan memperhatikan syarat data latih yang tercantum di NOTICE.md (sebagian berasal dari data bersyarat
non-komersial dan bagi-serupa). Tanpa lisensi tertulis, hak cipta tetap pada pemilik dan pihak lain tidak otomatis
boleh memakai atau menyebarkan ulang.
"""


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def unique_param_count(state_dict):
    """Jumlah parameter sebenarnya. state_dict memuat bobot yang dibagi pakai di beberapa kunci (mis. embedding token yang sama dengan
    lm_head); menjumlahkan semua kunci menghitungnya berkali-kali (151,2 juta padahal modelnya 100,9 juta). Di sini bobot yang berbagi
    penyimpanan, offset, bentuk, dan langkah yang sama dihitung sekali."""
    seen, total = set(), 0
    for t in state_dict.values():
        key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()))
        if key not in seen:
            seen.add(key)
            total += t.numel()
    return total


def model_card(v, n_params, step, size_mb):
    return f"""# Kartu Model: hai07-100m v{v}

Model bahasa Indonesia kecil (decoder-only, {n_params/1e6:.1f} juta parameter) dengan pembaca catatan untuk aplikasi **Memori**
(AI yang bisa diajari pemiliknya, jawaban selalu disertai sumber). Dilatih dari nol di kuota GPU gratis.

## Ringkasan
| | |
|---|---|
| Arsitektur | GPT decoder-only: 12 layer, d_model 768, 12 head (GQA 4 KV-head), SwiGLU, RMSNorm, posisi learned, konteks 256, weight tying |
| Parameter | {n_params:,} |
| Tokenizer | subword hibrida, kosakata 32.768 (dilatih dari korpus Indonesia) |
| Berkas bobot | `{MODEL_FILE.format(v=v)}` ({size_mb:.0f} MB, fp32, dapat dimuat dengan `weights_only=True`) |
| Titik latih | SFT tahap 5, step {step} (checkpoint terbaik menurut validasi) |
| Bahasa | Indonesia |
| Lisensi | bobot: CC BY-NC-SA 4.0 (non-komersial, bagi-serupa); kode: MIT. Lihat `LISENSI.txt` |

## Cara dilatih
1. **Pretraining** ±1,4 miliar token yang dilihat (korpus 2,05 miliar token: Wikipedia Indonesia, HPLT 2.0, FineWeb-2 ind_Latn);
   42.000 step; validation loss 2,17 (perplexity ±8,8).
2. **SFT bertahap** (±10 menit GPU per tahap): instruksi umum, QA berbasis teks, pembaca pada hasil pencarian,
   catatan memori sintetis, dan catatan rumit kompositional. Loss hanya dihitung pada jawaban.
Rincian data dan lisensinya: `NOTICE.md`.

## Penggunaan yang dimaksud
Pembaca untuk **memori lokal yang diajari pemilik**: tulis fakta pendek, tanyakan kembali, baca jawaban beserta sumbernya.
Model hanya membaca catatan dan menjawab dari situ; pagar pengaman hanya meloloskan jawaban yang berupa potongan catatan.

## Hasil pengukuran (soal ditahan; sampel kecil, gambaran kasar)
Pengaturan **Seimbang**. "Mengarang" = menjawab pertanyaan yang jawabannya tidak ada di catatan.

| Uji | Benar | Salah | "Belum tahu" | Mengarang |
|---|---|---|---|---|
| Fakta pendek terstruktur (memori 200 catatan, 100+50 soal) | ~74% | 0% | ~26% | 0% |
| Kalimat bebas sederhana (tulisan tangan, 20+8 soal) | ~50% | ~5% | ~45% | 0% |
| Kalimat rumit banyak fakta (tulisan tangan, 69+15 soal) | ~39% | ~14% | ~46% | ~7% |
| QA dari teks Wikipedia lewat pencarian (RAG, 144 soal) | ~10% | | | |

Skrip dan data uji ada di repositori sumber (`tools/*eval*.py`, `evalsets/`).

## Keterbatasan (penting)
- **Bukan sumber fakta.** Pengetahuan umum model lemah dan sering keliru; jangan dipakai tanpa catatan sumber.
- Pada kalimat rumit hanya sekitar separuh jawaban benar, bahkan saat diberi catatan yang tepat. Kalimat sederhana (satu fakta per kalimat) jauh lebih akurat.
- Lebih sering menjawab "belum tahu" daripada salah (disengaja); sebagian pertanyaan yang sebenarnya terjawab akan ditolak.
- Konteks hanya 256 token; catatan panjang tidak muat.
- Tidak ada pelatihan keselamatan atau penyelarasan. Data pretraining berasal dari web terbuka, sehingga model **dapat menghasilkan teks bias, keliru, atau tidak pantas** bila dipakai sebagai generator bebas. Pakailah lewat aplikasi Memori.
- Hanya bahasa Indonesia; kemampuan bahasa lain tidak diuji.
- Sebagian data SFT adalah terjemahan mesin dan/atau berasal dari keluaran model bahasa lain; kualitas dan kesalahan terjemahan dapat terbawa.

## Pelaporan
Hasil ini diukur pada set kecil; selisih beberapa poin persen bukan bukti. Laporkan temuan kesalahan melalui repositori sumber.
"""


def notice(v):
    return f"""# NOTICE: sumber data dan atribusi (hai07-100m v{v})

Model ini dilatih dari gabungan data berikut. Lisensi dicatat sesuai keterangan pada halaman masing-masing sumber saat diunduh;
**periksa sendiri sebelum menyebarkan atau memakai secara komersial.**

## Lisensi paket ini
Kode: MIT. Bobot model: CC BY-NC-SA 4.0 (lihat `LISENSI.txt`). Data latih tidak disertakan dan tunduk pada lisensinya masing-masing di bawah.

## Pretraining (teks)
| Sumber | Keterangan | Lisensi menurut sumber |
|---|---|---|
| Wikipedia bahasa Indonesia (idwiki, dump 2026-09-01) | Wikimedia Foundation dan kontributor | CC BY-SA 4.0 (dan GFDL) |
| HPLT 2.0 cleaned, bagian `ind_Latn` (`HPLT/HPLT2.0_cleaned`) | HPLT project; teks hasil web crawl | HPLT mencatat data sebagai CC0; hak cipta teks asal tetap pada pemiliknya |
| FineWeb-2, bagian `ind_Latn` | HuggingFaceFW; berasal dari Common Crawl | ODC-By 1.0 (dan ketentuan Common Crawl) |

## Fine-tuning (instruksi, QA)
| Sumber | Lisensi menurut sumber | Catatan |
|---|---|---|
| Ichsan2895/alpaca-gpt4-indonesian | CC BY-SA 4.0 | Terjemahan mesin dari Alpaca-GPT4 (jawaban dibuat GPT-4). Dataset asal Alpaca-GPT4 dirilis CC BY-NC 4.0 dan tunduk pada ketentuan penggunaan OpenAI. |
| cahya/alpaca-id | MIT | Terjemahan dari Stanford Alpaca (dibuat dengan text-davinci-003), yang dirilis CC BY-NC 4.0 dan tunduk pada ketentuan OpenAI. |
| Ichsan2895/OASST_Top1_Indonesian | CC BY-SA 4.0 | Terjemahan dari OpenAssistant oasst1 (Apache-2.0). |
| FreedomIntelligence/sharegpt-indonesian | Apache-2.0 | Berasal dari percakapan ShareGPT (keluaran ChatGPT). |
| TyDi QA (secondary task, bahasa Indonesia) | Apache-2.0 | Clark dkk., 2020. |
| IDK-MRC | CC BY 4.0 | Putri dan Oh, 2022. |
| Data sintetis catatan memori | dibuat oleh pemilik proyek | Dihasilkan program (`tools/memory_gen*.py`), tanpa teks pihak ketiga. |

## Yang perlu diperhatikan sebelum distribusi
- Sebagian data SFT berasal dari rantai yang berhulu pada dataset **non-komersial (CC BY-NC)** dan keluaran model bahasa komersial
  yang syaratnya melarang sebagian penggunaan. Karena itu bobot dirilis non-komersial (CC BY-NC-SA 4.0); **jangan menganggap model ini boleh dipakai secara komersial.**
- Data bagi-serupa (CC BY-SA) memunculkan pertanyaan hukum yang belum mapan tentang apakah bobot model turunan wajib berlisensi serupa.
  Pilihan non-komersial dan bagi-serupa dipakai sebagai langkah hati-hati.
- Versi komersial memerlukan pengulangan fine-tuning dengan data yang lisensinya bersih.
- Dokumen ini bukan nasihat hukum.
"""


def package_readme(v):
    return f"""# Memori v{v}: AI kecil yang bisa Anda ajari (offline)

Paket ini menjalankan model bahasa Indonesia 100 juta parameter di komputer Anda sendiri. Anda mengajarinya fakta dengan kalimat
biasa, lalu menanyakannya kembali; jawaban selalu disertai catatan sumber, dan jika tidak yakin ia menjawab "belum tahu".
Semua berjalan lokal: tidak ada data yang dikirim ke luar.

## Cara menjalankan
**Windows:** klik ganda `jalankan.bat`.  **Linux/macOS:** `bash jalankan.sh`.

Pertama kali, skrip menyiapkan lingkungan Python (perlu Python 3.10+ dan internet, sekitar 250 MB, hanya sekali).
Setelah itu peramban terbuka di `http://127.0.0.1:8765`. Tutup dengan Ctrl+C di jendela terminal.

Tanpa skrip: `pip install -r requirements.txt` lalu `python tools/memory_web.py --open`.
Versi baris perintah: `python tools/memory_chat.py`.

## Panduan
- Pemasangan dan pemecahan masalah: `PANDUAN_INSTALASI.md`.
- Mengajar AI: `PANDUAN_MENGAJAR.md` (singkat).

## Mulai mengajar
Ringkasnya: Intinya: **satu fakta per kalimat, sebut nama lengkap, tulis tanggal yang pasti**,
dan selalu periksa sumber yang ditampilkan di bawah jawaban.

## Catatan penting
- Model kecil: akurat pada kalimat sederhana (±74% pada uji terstruktur), hanya sekitar separuh pada kalimat rumit. Lihat `MODEL_CARD.md`.
- Catatan Anda tersimpan di `data/memory/memory.db`. Cadangkan berkas itu. Server hanya menerima koneksi dari komputer ini; jangan dibuka ke jaringan.
- Sumber data latih dan lisensinya: `NOTICE.md`. Status lisensi paket: `LISENSI.txt`.
- Integritas berkas: bandingkan dengan `SHA256SUMS.txt`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT_REPO)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--license-dir", default=None, help="Folder berisi LISENSI.txt dan teks lisensi (bawaan: licenses/).")
    args = ap.parse_args()
    v = args.version
    out_root = PROJECT_ROOT / args.out
    pkg = out_root / f"hai07-memori-v{v}"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "model").mkdir(parents=True)

    # 1) bobot saja
    src = PROJECT_ROOT / args.checkpoint
    payload = torch.load(str(src), map_location="cpu", weights_only=True)
    slim = {k: payload[k] for k in ("epoch", "step", "env", "metadata", "model_state_dict") if k in payload}
    model_path = pkg / MODEL_FILE.format(v=v)
    torch.save(slim, str(model_path))
    n_params = unique_param_count(slim["model_state_dict"])   # bobot yang dibagi pakai (weight tying) dihitung sekali
    # 2) verifikasi: berkas kecil dapat dimuat aman dan identik dengan sumbernya
    again = torch.load(str(model_path), map_location="cpu", weights_only=True)
    for k, t in payload["model_state_dict"].items():
        assert torch.equal(t, again["model_state_dict"][k]), f"bobot berbeda: {k}"
    print(f"[paket] bobot identik ({len(again['model_state_dict'])} tensor); {src.stat().st_size/1e6:.0f} MB -> {model_path.stat().st_size/1e6:.0f} MB")
    del payload

    # 3) kode jalan minimal
    for d in ("apps", "apps/modules"):
        (pkg / d).mkdir(parents=True, exist_ok=True)
        for f in (PROJECT_ROOT / d).glob("*.py"):
            shutil.copy2(f, pkg / d / f.name)
    (pkg / "tools").mkdir()
    default_ckpt_pkg = MODEL_FILE.format(v=v)
    for name in RUNTIME_TOOLS:
        text = (PROJECT_ROOT / "tools" / name).read_text(encoding="utf-8")
        text = text.replace(DEFAULT_CKPT_REPO, default_ckpt_pkg)
        (pkg / "tools" / name).write_text(text, encoding="utf-8", newline="\n")
    shutil.copy2(PROJECT_ROOT / "tools" / "kamus_bawaan.txt", pkg / "tools" / "kamus_bawaan.txt")
    (pkg / "web" / "memory").mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "web" / "memory" / "index.html", pkg / "web" / "memory" / "index.html")
    (pkg / "data" / "prod").mkdir(parents=True)
    for f in VOCAB_FILES:
        shutil.copy2(PROJECT_ROOT / "data" / "prod_v2" / f, pkg / "data" / "prod" / f)
    (pkg / "data" / "memory").mkdir(parents=True)
    (pkg / "data" / "memory" / ".gitkeep").write_text("", encoding="utf-8")

    # 4) dokumen, peluncur, kebutuhan
    from_card = model_card(v, 100_879_104, slim["step"], model_path.stat().st_size / 1e6)
    (pkg / "MODEL_CARD.md").write_text(from_card, encoding="utf-8", newline="\n")
    (pkg / "NOTICE.md").write_text(notice(v), encoding="utf-8", newline="\n")
    (pkg / "README.md").write_text(package_readme(v), encoding="utf-8", newline="\n")
    for guide in ("PANDUAN_MENGAJAR.md", "PANDUAN_INSTALASI.md"):
        shutil.copy2(PROJECT_ROOT / "docs" / guide, pkg / guide)
    (pkg / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8", newline="\n")
    (pkg / "jalankan.bat").write_text(BAT, encoding="utf-8", newline="\r\n")
    (pkg / "jalankan.sh").write_text(SH, encoding="utf-8", newline="\n")
    (pkg / "VERSION").write_text(v + "\n", encoding="utf-8")
    lic = PROJECT_ROOT / (args.license_dir or "licenses")
    if (lic / "LISENSI.txt").exists():
        for name in ("LISENSI.txt", "LICENSE-CODE-MIT.txt", "LICENSE-MODEL-CC-BY-NC-SA-4.0.txt"):
            shutil.copy2(lic / name, pkg / name)
    else:
        (pkg / "LISENSI.txt").write_text(NO_LICENSE, encoding="utf-8", newline="\n")
        print("[paket] PERINGATAN: folder licenses/ tidak ditemukan; paket dibuat tanpa lisensi.")

    # 5) checksum + zip
    lines = []
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{sha256(p)}  {p.relative_to(pkg).as_posix()}")
    (pkg / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    zip_path = out_root / f"hai07-memori-v{v}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(pkg.rglob("*")):
            if p.is_file():
                z.write(p, f"hai07-memori-v{v}/" + p.relative_to(pkg).as_posix())
    print(f"[paket] {pkg}  ({len(lines)} berkas)")
    print(f"[paket] {zip_path}  ({zip_path.stat().st_size/1e6:.0f} MB)  sha256={sha256(zip_path)}")


if __name__ == "__main__":
    main()
