# Router Online Dataset Builder v5

Builder ini menggunakan MASSIVE Bahasa Indonesia melalui Parquet hasil konversi Hugging Face. Ia **tidak** menjalankan `massive.py` dan tidak memakai Dataset Viewer API.

## Mengapa v5

Tes langsung pada lingkungan pengguna telah berhasil memuat ketiga file berikut dengan `load_dataset("parquet", data_files=...)`:

- `id-ID/train/0000.parquet` — 11.514 baris
- `id-ID/validation/0000.parquet` — 2.033 baris
- `id-ID/test/0000.parquet` — 2.974 baris

URL Parquet menggunakan branch `refs/convert/parquet`, sesuai tes `test_massive.py` yang berhasil. Kolom `intent` bertipe `ClassLabel`; builder memakai `features["intent"].int2str(...)` untuk memperoleh nama intent asli.

## Instalasi dan menjalankan (Windows PowerShell)

Aktifkan venv, masuk ke folder builder, lalu:

```powershell
python -m pip install -r requirements.txt
python -m py_compile build_online_datasets.py
python build_online_datasets.py --output generated_online
```

Tidak perlu menurunkan versi `datasets`. Tidak wajib mengatur `HF_TOKEN` untuk repository publik; tanpa token Hugging Face hanya memberi peringatan batas rate anonim.

Jika ingin menguji MASSIVE saja tanpa memuat INTISARI:

```powershell
python build_online_datasets.py --output generated_online --skip-intisari
```

## Kebijakan label (Opsi A)

Label router proyek: `urut`, `hitung`, `tanggal`, `cari`, `sapaan`, `tolak`.

- MASSIVE hanya dipetakan otomatis jika intent sumber ada di `MASSIVE_MAP`.
- Intent MASSIVE yang belum jelas **masuk ulasan manual**, bukan otomatis `tolak`.
- Tidak ada intent MASSIVE yang dipaksakan menjadi `urut` atau `tolak`.
- INTISARI hanya memberi label `sapaan` pada formula sosial pendek dan jelas. Semua pesan lainnya masuk ulasan manual.

## Pemisahan data

- `evalsets_online.jsonl`: hanya intent MASSIVE yang dipetakan dari split `test`.
- `router_tulisan_anda_online.jsonl`: hanya intent MASSIVE yang dipetakan dari `train` + `validation`, ditambah formula sosial INTISARI yang jelas.
- Pesan MASSIVE yang belum dipetakan dari semua split masuk review dan manifest. Baris dari split `test` tidak pernah dimasukkan ke file training router.

## Output

1. `evalsets_online.jsonl` — format `{"text":"...","label":"..."}`.
2. `router_tulisan_anda_online.jsonl` — format `{"message":"...","label":"...","family":"...","salah_ketik":false}`.
3. `ulasan_label_online.md` — format empat kolom sesuai template: `No | Pesan | Label | Benar?`. Isi label yang kosong dengan salah satu dari enam label proyek.
4. `dataset_sources_manifest.csv` — provenance lengkap, label intent sumber, split, mapping, dan alasan review.
5. `dataset_summary.json` — jumlah data dan distribusi label/sumber.

Markdown review dibatasi 10.000 baris secara default agar nyaman dibuka. Manifest CSV tetap memuat seluruh data review. Ubah batas dengan `--max-review-rows 20000` atau `--max-review-rows 0` untuk tidak menampilkan baris review di Markdown (manifest tetap lengkap).

## Lisensi dan atribusi

MASSIVE mencantumkan lisensi CC BY 4.0. Periksa ketentuan dataset INTISARI pada dataset card sebelum mendistribusikan hasil gabungan. Simpan informasi sumber dan atribusi.
