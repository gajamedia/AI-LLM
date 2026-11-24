# hai07-ocd

Pipeline LLM gaya GPT (decoder-only) untuk teks berbahasa Indonesia:
**korpus → tokenizer subword hibrida 32.768 → pretokenisasi biner → training → generasi**.

Semua kode berjalan di **CPU-only** (diukur pada mesin ini) dan diuji otomatis dengan pytest.

---

## Status

| Tahap | Isi | Status |
|---|---|---|
| 1A–1F | Korpus (Wikipedia ID + HPLT), tokenizer 32.768, pretokenisasi | **Selesai** |
| 2A–2C | Koreksi konfigurasi, RMSNorm/CE numerik, scheduler LR | **Selesai** |
| 3 | KV cache + `position_offset` benar pada generasi | **Selesai** |
| 4 | Hygiene repo (.gitignore, README, pyproject, pytest) | **Selesai** |
| — | Training penuh 421M token | **Selesai di GPU** (model 100M, Kaggle; lihat [Training GPU di Kaggle](#training-gpu-di-kaggle-model-100m)). CPU tidak layak. |
| — | Pretraining v2 (100M, 42.000 step) | **Selesai** (val loss 2,17; baseline 2,47) |
| — | SFT bertahap 1–5 (instruksi, QA berbasis teks, hasil pencarian, catatan memori) | **Selesai**; lihat [Memori yang bisa diajari](#memori-yang-bisa-diajari) |
| — | Aplikasi memori (web + CLI) | **Prototipe berfungsi**; akurasi terbatas, lihat [panduan](docs/PANDUAN_MENGAJAR.md) |

Arsitektur model (production, tidak berubah): 8 layer, `d_model=384`, GQA 6 head / 3 KV head,
SwiGLU (hidden 1024), RMSNorm, posisi learned, konteks 256, vocab 32.768,
`tie_weights=True` → **25.663.872 parameter (25.66M)**.

---

## Persyaratan

- Python 3.10 (dikembangkan pada 3.10.11, Windows)
- `torch` build CPU maupun CUDA — perilaku terukur ada di
  [Batas perangkat keras](#batas-perangkat-keras)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # + pytest
```

> `tools/requirements-hf-corpus.txt` memaksa `datasets==2.19.2` sementara env ini
> memakai `datasets==5.0.1`. **Jangan** menginstallnya ke env dasar tanpa venv terpisah.

---

## Struktur repo

```
apps/        model & utilitas inti (bukan package; ditambahkan ke sys.path)
  config.py            DevelopmentConfig / ProductionConfig / TestingConfig
  gptModel.py          GPTModel (embedding, stack, LM head, weight tying)
  transformerStack.py  TransformerStack + KV cache per layer
  transformerBlock.py  attention + MLP (SwiGLU / FeedForward) + norm
  attention.py         GQA, RoPE opsional, causal mask, KV cache
  modules/rmsnorm.py   RMSNorm dengan akumulasi fp32
  embedding.py         token + positional embedding
  tokenizer2.py        HybridTokenizerLLM
  dataset2.py          packed/sliding token dataset
tools/       skrip CLI (korpus, tokenizer, training, generasi)
tests/       pytest
data/        artefak (~17 GB, TIDAK di-commit; lihat .gitignore)
reports/     laporan JSON + checkpoint + log
```

---

## Alur kerja

> **Berkas besar yang sengaja tidak disimpan di komputer ini** (dihapus untuk menghemat ruang; semuanya bukan bagian repo git): checkpoint SFT tahap 1 sampai 4 (`reports/checkpoints_100m_sft*`, selain `sft_mem2`), checkpoint penuh v1 (bobotnya masih ada di `reports/checkpoints_100m/gpt_best_weights.pt`), dump Wikipedia mentah (`data/corpus_dump_20260901`, unduh ulang dengan perintah di bawah), dan data RAG (`data/rag`, dibuat ulang dengan `tools/rag_index.py` setelah dump tersedia). Alat percobaan RAG (`tools/rag_eval.py`, yang bawaannya memakai checkpoint SFT tahap 2, dan `tools/build_sft_rag_data.py`) memerlukan berkas-berkas itu dipulihkan (checkpoint tahap lama ada di keluaran notebook Kaggle). Aplikasi Memori hanya memakai `reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt` dan kosakata di `data/prod_v2/`.

Semua perintah di biasakan dijalankan dari root repo. Di Windows gunakan
`venv\Scripts\python.exe` bila virtualenv tidak aktif.

### 1. Korpus

**Wikipedia Indonesia** (sumber utama):

```bash
python tools/download_wikimedia_dump.py --wiki idwiki --snapshot 20260901 \
    --kind articles --output-dir data/corpus_dump_20260901/raw

# --resume aman: melanjutkan dari sidecar .progress.json dan menghasilkan
# byte yang identik dengan run penuh.
python tools/parse_wikimedia_dump.py --wiki idwiki --snapshot 20260901 \
    --input  data/corpus_dump_20260901/raw/idwiki-20260901-pages-articles-multistream.xml.bz2 \
    --output data/corpus_dump_20260901/wiki.jsonl --resume
```

**HuggingFace HPLT (opsional, menambah ~51M token):**

```bash
python tools/import_hf_corpus.py --dataset HPLT/HPLT2.0_cleaned --split train \
    --license cc0-1.0 --max-rows 8000 --output data/hf_hplt/hplt_all.jsonl
python tools/filter_hf_corpus.py --input data/hf_hplt/hplt_all.jsonl \
    --output data/hf_hplt/hplt_clean.jsonl --min-indonesian-stopwords 2
```

**Bersihkan + split:**

```bash
python tools/clean_jsonl_corpus.py \
    --input data/corpus_dump_20260901/wiki.jsonl \
    --input data/hf_hplt/hplt_clean.jsonl \
    --output data/corpus_clean.jsonl

# split deterministik per-hash sumber (source overlap = 0),
# sekaligus mengirim teks & sampel untuk training tokenizer
python tools/split_corpus_by_source_hash.py --input data/corpus_clean.jsonl \
    --output-dir data/split --validation-every 50 \
    --emit-text --tokenizer-sample 1500000 --tokenizer-sample-seed 17

python tools/corpus_stats.py data/corpus_clean.jsonl \
    --tokens-per-character 0.288262 --output reports/corpus_stats_clean.json
```

Hasil saat ini: **8.381.399 baris / 1.432.530.596 karakter / ±412,9M token estimasi**.

> ⚠️ Aturan keras:
> - `filter_hf_corpus.py` **hanya** untuk output `import_hf_*`. Pada baris Wikipedia ia
>   menimpa `source_id` jadi `hf:None:...`.
> - `merge_corpora.py` membuka output dengan mode `"w"` — **jangan merge in-place**.
> - Wikimedia menolak request tanpa User-Agent (HTTP 403).
> - `corpus_stats.py` memakai argumen **posisional**, bukan `--input`.

### 2. Tokenizer

`tools/train_tokenizer_progress.py` membaca korpus sebagai **teks baris biasa, bukan JSONL**,
jadi harus diberi file `.txt`.

```bash
python tools/train_tokenizer_progress.py --env production --force \
    --corpus-path data/split/tokenizer_sample.txt \
    --output-dir data/prod --max-vocab-samples 1500000
```

Menghasilkan `data/prod/{vocab,token2id,id2token}_prod.json` (32.768 token).

```bash
python tools/evaluate_tokenizer_corpus.py --env production --sample-size 2000 \
    --train data/split/train.jsonl --validation data/split/validation.jsonl \
    --tokenizer-dir data/prod --output reports/evaluate_tokenizer_prod.json
```

Hasil: `roundtrip_exact_rate = 1.0`, `unknown_token_rate ≈ 0.003`,
`tokens_per_character ≈ 0.279` (rata-rata global terukur **0.2883**).

### 3. Pretokenisasi

```bash
python tools/dataset_pretokenizer_fast.py --env production \
    --train data/split/train.txt --validation data/split/validation.txt \
    --tokenizer-dir data/prod --output-dir data/prod --log-interval 500000
```

Menghasilkan `train_tokens.bin` (420.999.317 token), `validation_tokens.bin`
(8.701.871 token), index `.docs.npy`, dan `pretokenization_manifest.json`.
Encoder trie diuji setara dengan DP referensi `tokenizer2`.

### 4. Training

```bash
python tools/train_gptV2.py --env production \
    --token-file data/prod/train_tokens.bin \
    --scheduler cosine --warmup-steps 500 --min-lr 2e-5 --no-amp \
    --save-every 50 --log-every 20 \
    --val-every 100 --val-token-file data/prod/validation_tokens.bin
```

Lanjutkan dari checkpoint (berlaku juga untuk `--scheduler`/`--lr` AdamW):

```bash
python tools/train_gptV2.py --env production \
    --token-file data/prod/train_tokens.bin \
    --resume reports/checkpoints/gpt_last.pt ...
```

- `--max-steps` bersifat **kumulatif** terhadap `global_step` hasil resume
  (resume dari step 6 dengan `--max-steps 8` berhenti di total 8).
- `--scheduler {cosine,plateau,none}`, `--warmup-steps`, `--min-lr` menimpa config;
  `--plateau` / `--no-plateau` tetap tersedia untuk kompatibilitas lama.
- `--amp` hanya efektif pada CUDA. Pada CPU, `bf16` autocast terbukti **>70× lebih lambat**
  sehingga `TRAIN_AMP_ENABLED=False` adalah default yang benar untuk mesin ini.

### 5. Generasi

```bash
python tools/generate_gpt.py --env production \
    --checkpoint reports/checkpoints/gpt_last.pt \
    --prompt "Indonesia adalah" --max-new-tokens 60 --seed 7
```

- KV cache **aktif by default** (±3.8–4.6× lebih cepat pada pengukuran).
  `--no-cache` tersedia untuk debug/perbandingan.
- Prompt lebih panjang dari konteks dipotong ke 256 token terakhir **dan dilaporkan**.
- Generasi berhenti saat konteks penuh (embedding posisi learned tidak punya slot lagi).

---

## Training GPU di Kaggle (model 100M)

Config `--env production_100m`: d_model 768, 12 layer, GQA 12/4 head, **100.879.104 parameter**.
Tokenizer & data sama dengan production; checkpoint **tidak kompatibel** dengan production (25M).

- Notebook: `kaggle/hai07_train_100m.ipynb` (T4, fp16 autocast + GradScaler otomatis bila GPU < compute capability 8.0).
- Input Kaggle: dataset `hai07-prod-data` (isi `data/prod/`), `hai07-code` (zip `apps/` + `tools/`), dan `hai07-ckpt` untuk sesi lanjutan.
- Data **wajib disalin ke disk lokal** sebelum training. Membaca `np.memmap` langsung dari `/kaggle/input`
  mengakibatkan ~18,7 s/step; dengan salinan lokal ~2,1 s/step (~15 ribu token/s).
- Resume memakai nomor step terbesar; `--max-steps` kumulatif dan jadwal cosine bergantung padanya, jangan diubah antar sesi.
- Hasil run (25.600 step, 2 epoch, ~15 jam GPU): validation loss **3,275** (ppl ~26,4), vs 4,56 pada model 25M.
  Kalimat lancar tetapi fakta sering salah (rasio ~4 token/parameter).
- Unduhan Kaggle bisa mengekstrak file `.pt` menjadi folder; kemas ulang dengan zip tanpa kompresi (prefix folder `gpt_best/`).

---

## Memori yang bisa diajari

Model tetap (bobot tidak berubah); pengetahuan ada di berkas memori lokal yang bisa diajari pemiliknya
dengan kalimat biasa. Setiap jawaban menampilkan catatan sumber, dan jika tidak yakin sistem menjawab "belum tahu".
Panduan pengguna: [instalasi](docs/PANDUAN_INSTALASI.md) dan [mengajar](docs/PANDUAN_MENGAJAR.md).

```bash
python tools/memory_web.py --open      # web: http://127.0.0.1:8765  (hanya localhost, tanpa dependensi tambahan)
python tools/memory_chat.py            # CLI: ingat: ... / pertanyaan? / lupakan N / daftar
```

Komponen: `tools/memory_agent.py` (penyimpan FTS5, pembaca, pagar pengaman), `tools/memory_web.py` + `web/memory/index.html`
(antarmuka), `tools/rag_index.py` (indeks Wikipedia, opsional). Model bawaan: `reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt`
(berkas model tidak ada di git; lihat bagian Training GPU).

Rangkaian SFT (semua dilatih di Kaggle T4, masing-masing ±10 menit; skrip data di `tools/build_sft_*_data.py`, pelatih `tools/train_sft.py`,
notebook `kaggle/hai07_sft_*_100m.ipynb`):
1. instruksi umum (data terjemahan Indonesia), 2. QA berbasis teks (TyDi QA + IDK-MRC), 3. pembaca pada hasil pencarian BM25,
4. catatan memori sintetis, 5. catatan rumit kompositional.

Set uji tulisan tangan yang dibekukan ada di `evalsets/` (jangan dipakai menyetel data atau model). Hasil pengukuran (soal ditahan; skrip `tools/eval_qa.py`, `rag_eval.py`, `memory_*eval*.py`; angka di `reports/eval_*.json`):

| Uji | Hasil |
|---|---|
| QA dari teks Wikipedia (RAG ujung ke ujung, 144 soal) | ~10% benar (BM25 menemukan jawaban di top-3 hanya pada ~45% soal); **belum berguna** |
| Catatan pendek terstruktur, memori 200 catatan (agen bawaan) | ~74% benar, 0% salah, 0% mengarang |
| Kalimat bebas sederhana tulisan tangan (20 soal) | ~50% benar, ~5% salah |
| Kalimat rumit banyak fakta (69 soal, set dibekukan) | ~39% benar, ~14% salah, ~46% "belum tahu" |
| Impor dokumen .txt/.md (100 soal Wikipedia) | Pencarian menemukan bagian yang benar di 3 teratas pada 94% soal; jawaban AI benar hanya 13% (fitur menampilkan bagian dokumen, jawaban AI hanya dugaan) |

Keterbatasan yang diketahui: model 100M tidak andal sebagai sumber fakta; pada kalimat rumit hanya sekitar separuh benar bahkan
saat diberi catatan yang tepat; set uji kecil (puluhan soal) sehingga selisih beberapa poin bukan bukti. Data latih memuat Wikipedia (CC-BY-SA),
dataset terjemahan CC-BY-SA, dan Alpaca-GPT4 (jawaban dibuat model OpenAI): periksa lisensi sebelum mendistribusikan model.

### Pemilih alat buatan sendiri (opsional, bawaan MATI)
`tools/memory_router.py` + `tools/router_weights.npz` (0,5 MB, numpy saja, berjalan di CPU): classifier kecil yang **mengusulkan** alat (urut, hitung, tanggal, tolak, sapaan)
untuk pesan yang lolos dari aturan ketat; alat deterministik tetap harus berhasil membaca angka/tanggalnya, jika tidak pesan jatuh ke pencarian catatan.
Pengaturan lewat berkas: salin `memori.env.example` menjadi `memori.env` (di folder utama), ubah nilainya, lalu mulai ulang aplikasi; variabel lingkungan sungguhan selalu mengalahkan berkas. Atau nyalakan dengan variabel lingkungan `MEMORI_ROUTER=on` (`shadow` = jalan paralel tanpa mengubah hasil; `MEMORI_ROUTER_LOG=berkas` mencatat hanya kelas dan jumlah kata, tidak menyimpan teks pesan).
Terukur pada set buatan pengembang (validasi 2.100 pesan): pesan alat yang terbaca naik dari 169 ke 293 dari 300 (urut), 78 ke 254 (hitung), 203 ke 294 (tanggal), pertanyaan catatan dibajak 0 dari 600.
**Belum diuji pada gaya tulisan pemilik lain**, karena itu bawaannya mati. Juga ada perbaikan jawaban yang terpotong di tengah kata ("Parangtriti" menjadi "Parangtritis"; tidak mengubah hasil model sekarang pada set uji beku).

### Pencarian hibrida BM25 + vektor (opsional, bawaan MATI)
`tools/hybrid_search.py` menggabungkan pencarian BM25 dengan pencarian vektor memakai Reciprocal Rank Fusion. Aktifkan lewat `memori.env` (`MEMORI_HYBRID=on`); bila embedder tidak bisa dimuat, aplikasi diam-diam kembali ke BM25. Vektor catatan disimpan di tabel `note_vec` (SQLite yang sama) dan hanya catatan baru yang dikodekan.
Embedder bawaan: `intfloat/multilingual-e5-small` (model orang lain; butuh `pip install torch transformers` dan unduhan ±470 MB sekali). Ini BUKAN model buatan sendiri dan menambah dependensi besar, karena itu tidak diaktifkan otomatis.
Terukur dengan `tools/hybrid_eval.py` (paragraf Wikipedia dari soal TyDi QA / IDK-MRC tahan uji di antara 3.761 paragraf; soal asli, tanpa data karangan; hasil di `reports/eval_hybrid_search.json`):

| Metode (284 soal) | hit@1 | hit@3 | hit@10 |
|---|---|---|---|
| BM25 (yang dipakai sekarang) | 68,3 | 81,3 | 89,4 |
| Vektor n-gram buatan numpy (tanpa model) | 47,5 | 66,9 | 83,1 |
| Hibrida BM25 + n-gram | 57,7 | 75,0 | 88,7 |
| Embedding E5-small saja | **83,1** | **92,3** | **95,1** |
| Hibrida BM25 + E5-small | 77,5 | 89,4 | 94,0 |

Batas kemiripan: vektor selalu mengembalikan "yang terdekat" walau tak relevan, sehingga bagian dokumen tak berhubungan ikut tampil. Karena itu hasil vektor hanya dipakai bila kemiripannya >= `MEMORI_HYBRID_MIN_SIM` (bawaan 0,86). Ukur: kemiripan E5 sangat rapat (paragraf benar median 0,887; "terdekat tapi salah" median 0,839; pertanyaan di luar topik 0,82-0,85), jadi pada 0,86 pertanyaan di luar topik 0/8 lolos tetapi hanya 77,8% paragraf benar yang lolos; sinonim yang lemah (mis. "minuman ... pantry" untuk catatan kopi di dapur) bisa tidak ditemukan. Turunkan nilainya untuk lebih banyak hasil dengan risiko bagian tak relevan.

Catatan pendek (85 catatan, 124 soal): BM25 sudah 96,0% hit@1; E5 98,4%; hibrida 99,2% (selisih 1-3 soal, bukan bukti). Kesimpulan jujur: vektor n-gram tanpa model terlatih **tidak membantu** (tidak diaktifkan); embedding terlatih membantu jelas pada paragraf panjang; pada set ini E5 saja lebih baik daripada hibrida RRF berbobot sama (RRF menarik hasil bagus ke bawah), tetapi hibrida lebih aman bila pertanyaan memuat nama/angka persis. Ukuran ini hanya menilai PENCARIAN; mutu jawaban model 100M tidak otomatis ikut naik. Set uji sama dipakai memilih RRF vs bobot, jadi angka tidak independen.

### Cadangan untuk pertanyaan "KATEGORI mana/apa ..." (aktif otomatis)
Model 100M sering menjawab "belum tahu" untuk pertanyaan seperti "tim mana yang rapat di ruang Anggrek?" (ia menjawab nama ruangnya, bukan nama tim, lalu ditolak penjaga). Hanya bila model tidak menjawab, `category_lookup` mengambil kata PERSIS yang mengikuti kategori di catatan ("Rapat tim produk ... di ruang Anggrek" -> "produk"). Alat berhenti dan tetap "belum tahu" bila ruang/entitas yang ditanya tidak ada di catatan, ada dua jawaban berbeda, atau tidak ada kata pembeda antar catatan; jawaban model tidak pernah ditimpa.
Terukur dengan model asli pada aplikasi utuh: kasus bertema 20 -> 23 dari 26 lolos, nama tim 0 -> 3 dari 3, jawaban salah dan mengarang tidak bertambah. Ini kemampuan APLIKASI, bukan model: uji kelulusan model mematikannya secara bawaan (`--app-fallbacks` untuk mengukur aplikasi utuh). Hanya menutup pola "kategori mana/apa"; pola lain (mis. "siapa yang ...") belum.

---

## Lisensi

- **Kode** (`apps/`, `tools/`, `web/`, notebook): [MIT](LICENSE).
- **Bobot model** yang dirilis (paket `hai07-memori`): **CC BY-NC-SA 4.0**, non-komersial dan bagi-serupa
  ([teks](licenses/LICENSE-MODEL-CC-BY-NC-SA-4.0.txt)). Pilihan hati-hati karena sebagian data fine-tuning berhulu pada sumber
  non-komersial; ringkasan dan alasan di [`licenses/LISENSI.txt`](licenses/LISENSI.txt), sumber data di paket (`NOTICE.md`).
- Data latih tidak ada di repositori dan tunduk pada lisensi sumbernya. Ini bukan nasihat hukum.

Membuat paket rilis: `python tools/package_release.py --version 1.1126.12` (keluaran di `dist/`, tidak masuk git).

---

## Konfigurasi

Lingkungan: `development` | `production` | `testing` — dipilih dengan `--env`
(pada `generate_gpt.py`, `--env` bisa di-omit bila checkpoint menyimpan env-nya).

Sebagian besar hyperparameter bisa ditimpa dari CLI (`--lr`, `--batch-size`,
`--block-size`, `--grad-accum`, `--epochs`, `--max-steps`, `--seed`, dst.).
Lihat `python tools/train_gptV2.py --help`.

Nilai penting (production): `TRAIN_LR=2e-4`, `DATASET_BATCH_SIZE=16`,
`DATASET_BLOCK_SIZE=256`, `TRAIN_GRAD_ACCUM=8`, `TRAIN_MAX_STEPS=40000`,
`TRAIN_SCHEDULER=cosine`, `TRAIN_WARMUP_STEPS=500`, `TRAIN_MIN_LR=2e-5`.

---

## Batas perangkat keras

Pengukuran pada mesin ini (8 CPU, `torch 2.14.0+cpu`, `torch.cuda.is_available() == False`):

| Metrik | Nilai |
|---|---|
| Forward 4.5 s + backward 8.3 s per micro-step (B=16, T=256) | **±12,8 s → ~320 tok/s** |
| Thread 1 / 2 / 4 / 8 | 26,1 s / 15,8 s / 15,5 s / **11,5 s (~355 tok/s)** |
| `bf16` autocast di CPU | **>70× lebih lambat** (2 micro-step tidak selesai dalam 900 s) |
| Rasio token/param (421M / 25,66M) | **16,4 token/param** (Chinchilla ideal ~20) |
| 1 epoch 421M token (±355 tok/s) | **±13–14 hari** |
| `TRAIN_MAX_STEPS=40000` optimizer step (8 × 16 × 256 = 32.768 token/step) | **1.310.720.000 token ≈ 43 hari** |

Satu optimizer step = `grad_accum × batch × block` = 8 × 16 × 256 = 32.768 token.
`--max-steps` menghitung **optimizer step**, bukan micro-step.

Konsekuensi: sebelum menjalankan training penuh, jalankan dulu **subset**
(50–100M token) untuk memvalidasi pipeline. `TRAIN_SAVE_EVERY=2000` setara ±51 jam
antara checkpoint — turunkan dengan `--save-every 50` (±76 menit) untuk run berbasis CPU.

---

## Test

```bash
python -m pytest
```

38 test menutup: RMSNorm fp32/dtype, plumbing konfigurasi (cfg menang atas
global `CONFIG`), SwiGLU `mlp_ratio`, keseimbangan forward/backward,
weight tying, matematika scheduler warmup+cosine, KV cache (setara dengan
tanpa cache, batas konteks, potong prompt), dan roundtrip tokenizer produksi.

---

## Masalah config yang belum diperbaiki

Tiga penyimpangan `ProductionConfig` yang diketahui — belum diubah agar tidak
menyembunyikan perilaku lama:

1. `DATASET_TOKEN_FILE = data/prod/tokens_prod.bin` tidak ada;
   yang dihasilkan adalah `train_tokens.bin` / `validation_tokens.bin`
   → wajib `--token-file`.
2. `TRAIN_VAL_TOKEN_FILE = None` (validasi opt-in lewat `--val-token-file`).
3. `CORPUS_FILE = data/corpus.txt` tidak ada; `REQUIRE_CORPUS_FILE = False` sehingga tidak error.
