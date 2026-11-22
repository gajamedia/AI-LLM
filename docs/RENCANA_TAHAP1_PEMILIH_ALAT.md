# Rencana tahap 1: pemilih alat buatan sendiri

Status: **langkah 1 (data) selesai**; pelatihan dan perubahan kode aplikasi masih menunggu hasil uji penuh Qwen 1,5B dan 3B di Kaggle (agar target tidak disetel ke arah yang keliru).
Sudah ada: `tools/make_router_data.py`, set validasi/jebakan/berisik di `evalsets/`, data latih di `data/router/` (dibuat ulang oleh skrip), berkas ulasan label, dan templat pesan tulisan pemilik proyek.
Set uji dan kriteria yang sudah disetujui (`evalsets/memory_router*.jsonl`, `evalsets/acceptance_criteria.json`) **tidak diubah** oleh rencana ini.

## 1. Tujuan dan batasan
- **Tujuan:** memilih satu dari 6 kelas untuk tiap pesan pemakai: `urut`, `hitung`, `tanggal`, `cari` (tanya catatan), `sapaan`, `tolak` (permintaan menulis, menerjemah, merangkum). Dibuat sendiri, kecil, berjalan di CPU, tanpa model bahasa besar.
- **Bukan tujuan:** menggantikan alat deterministik (hitung, urut, tanggal tetap kode), atau mengubah pembaca catatan.
- **Garis dasar:** aturan di kode 95–100% pada tiga set buatan pengembang tetapi hanya 43% pada parafrase sebelum diperluas; model 100M hanya 15–18% (setara menebak). Aturan bagus pada gaya yang sudah kita tulis; kelemahannya ada pada gaya yang belum terbayang.

## 2. Temuan audit set pesan (197 pesan, baca-saja)
| Temuan | Arti bagi rancangan |
|---|---|
| Kelas seimbang (12/10/10 per kelas, `cari` 12/10/15) | Cukup untuk ukur, tetapi tiap set hanya 10 sampai 15 pesan per kelas: selang kepercayaan 95% pada 90% sekitar ±8 poin. **Perlu set validasi yang jauh lebih besar** |
| 2 pesan identik antar set (`hai`, `assalamualaikum`) dan 6 sampai 7 pesan hampir sama per set (kemiripan > 0,6, hampir semuanya sapaan) | Kebocoran kecil; set sapaan lebih mudah dari yang tampak. Data latih harus dibersihkan terhadap set uji |
| Panjang pesan berbeda per kelas (`sapaan` median 2 kata, `urut` 7, `tanggal` 9) | Classifier bisa **mengambil jalan pintas lewat panjang kalimat**. Data latih harus menyeimbangkan panjang antar kelas |
| Hanya 7 pesan memuat kata bilangan, 9 diawali "tolong/mohon/coba", **0 salah ketik**, 34 bertanda tanya | Gaya sempit. **Pemakai nyata akan berbeda**; classifier wajib diuji pada varian berisik (salah ketik, huruf, tanda baca) |
| 18 pesan memuat dua isyarat kelas atau `cari` yang berisi isyarat alat (mis. "Siapa yang membuat laporan bulanan?", "Berapa jumlah anggota koperasi?") | Inilah **pesan jebakan** yang paling penting; jumlahnya baru belasan, perlu ratusan |
| **Semua pesan ditulis oleh satu penulis (saya)** | **Risiko validitas terbesar.** Bila data latih juga saya tulis, classifier belajar gaya saya dan angka uji menjadi terlalu bagus. Perlu sumber gaya independen: pesan tulisan Anda dan pesan nyata |

## 3. Keputusan rancangan utama: classifier mengusulkan, alat memverifikasi
Biaya kesalahan tidak simetris:
- **Berbahaya:** pertanyaan tentang catatan salah dialihkan ke `sapaan` atau `tolak` ("Halo!" atau "tidak bisa menulis" sebagai ganti jawaban), atau ke alat yang menghasilkan sesuatu yang tidak diminta.
- **Ringan:** pesan alat diperlakukan sebagai `cari` (hasil "belum tahu").

Kebijakan:
1. Kelas `urut`, `hitung`, `tanggal` hanya dipakai bila alat deterministik yang sesuai **benar-benar menghasilkan jawaban**. Bila alat mengembalikan `None` (tidak ada dua angka, bukan ekspresi, tidak ada tanggal), pesan jatuh ke `cari`.
2. Kelas non-`cari` memerlukan probabilitas ≥ τ (ditentukan di set validasi, bukan di set uji). Di bawah itu, `cari`.
3. `sapaan` dan `tolak` memerlukan ambang lebih tinggi dan tetap harus pendek atau berbentuk perintah. Aturan lama dipertahankan sebagai **pengaman** selama masa peluncuran bertahap.
4. **Mode bayangan dulu** (lihat bagian 7): classifier dijalankan paralel tanpa mengubah perilaku, perbedaan dengan aturan dicatat, baru dinyalakan setelah ditinjau.

## 4. Classifier
Pilihan, dari yang paling sederhana:
| Pilihan | Parameter | Catatan |
|---|---|---|
| **A. Regresi logistik multikelas pada n-gram hash** (kata 1–2 gram + karakter 2–5 gram, 2^18 fitur) | ~1,6 juta | Paling jelas dan paling murah. Bobotnya bisa dibaca: fitur apa yang mendorong kelas apa. Latih di CPU dalam hitungan menit |
| B. Gaya fastText (embedding hash, rata-rata, lapisan linear) | ~8–16 juta | Menangkap kombinasi sedikit lebih baik; dibuat bila A tidak lulus |
| C. Model 100M sendiri + kepala klasifikasi | ~100 juta | Terlalu berat untuk tugas ini; hanya bila A dan B gagal |

**Rekomendasi:** mulai dari A sebagai garis dasar, naik ke B hanya bila A gagal pintu lulus. Normalisasi masukan: huruf kecil, hapus aksen, angka menjadi token `<angka>` dengan bentuknya, pola tanggal menjadi `<tanggal>`, kata bilangan menjadi `<bilangan>`. Latih dengan bobot kelas dan penghalusan label. Hanya `numpy` dan `torch` yang sudah ada, tanpa ketergantungan baru. Berkas model kecil (cukup beberapa MB).

## 5. Data
Semua data **buatan sendiri** (bebas masalah lisensi). Sumber:
- **Pembangkit pesan berbasis templat** per kelas, dengan variasi: ≥ 15 sinonim kata kerja per alat (termasuk gaya santai seperti "diurutin"), format angka (digit, kata bilangan, desimal, pemisah ribuan, "Rp"), format tanggal, awalan "tolong/mohon/coba/bisa", akhiran "ya/dong/nih", huruf kecil semua, tanpa tanda baca, campur bahasa Inggris, **salah ketik** (3–5% huruf).
- **`cari`:** pertanyaan dari pembangkit fakta yang sudah ada (`tools/memory_gen.py`, ragam pertanyaan sangat luas) dan soal pembaca di set uji beku, ditambah **pesan jebakan buatan** (≥ 500) yang memuat kata pemicu dalam pertanyaan biasa ("Siapa yang menyusun…?", "Berapa total biaya…?", "Hari apa … libur?", "Apa arti kata …?").
- **Tulisan Anda:** ±100 pesan bebas (urut/hitung/tanggal/tanya catatan/sapaan/tolak) yang Anda ketik tanpa melihat daftar saya. Dipakai sebagai set uji tambahan, **bukan** untuk melatih, supaya independen dari gaya saya.
- **Pesan nyata** dari pemakaian (bagian 7), diberi label oleh Anda, bila bersedia.

Pembagian dan kontrol kebocoran:
| Bagian | Isi | Dipakai untuk |
|---|---|---|
| train | templat keluarga A, ±3.000 per kelas (`cari` ±10.000) | melatih |
| validasi | **keluarga templat B yang tidak ada di train**, n ≥ 1.500 | memilih model dan ambang τ |
| uji resmi | tiga set yang sudah ada (197 pesan) | pintu lulus akhir, **tidak dipakai menyetel** |
| jebakan | ≥ 120 pertanyaan `cari` berisi kata pemicu, ditulis terpisah | ukur pembajakan |
| berisik | varian salah ketik/huruf/tanda baca dari set uji (otomatis) | ukur ketahanan |
| tulisan Anda | ±100 pesan | ukur independensi gaya |
| nyata | pesan pemakai berlabel | ukur di lapangan |

Kontrol: buang dari train semua pesan yang identik atau hampir sama (3-gram karakter, kemiripan ≥ 0,8) dengan pesan di set uji (ambang 0,8, bukan 0,6: templat memang boleh tumpang tindih, itu justru yang diukur set validasi; yang dilarang adalah pesan yang nyaris sama); kuota salah ketik 25% per kelas; seimbangkan panjang pesan antar kelas; **Anda memeriksa 150 sampel acak** (target ketidaksepakatan label ≤ 2%).

## 6. Cara mengukur dan pintu lulus
Tahap 1 lulus bila **semua** terpenuhi:
| Pintu | Syarat | Sumber |
|---|---|---|
| G1 (sudah disetujui) | akurasi ≥ 90% di tiap tiga set uji resmi, 0 pesan `cari` dibajak | `acceptance_criteria.json` |
| G2 | F1 makro ≥ 0,95 di set validasi (templat tak terlihat) | validasi |
| G3 | pesan jebakan salah dialihkan ≤ 1%; `cari` dibajak ≤ 0,5% pada seluruh pesan `cari` di validasi dan uji | jebakan, validasi |
| G4 | penurunan akurasi pada varian berisik ≤ 5 poin dari versi bersih | berisik |
| G5 | akurasi ≥ 85% pada tulisan Anda (gaya independen) dan tidak ada kemunduran pada qa beku dan 26 kasus | tulisan Anda, uji kelulusan penuh |
| G6 | ukuran ≤ 10 MB, latensi p95 ≤ 20 ms di CPU | ukur |
Semua angka dilaporkan dengan **selang kepercayaan 95% (Wilson)**. G5 ambang 85% adalah usulan awal: pesan gaya independen sengaja diberi syarat sedikit lebih longgar, dan saya akan menyesuaikannya bila hasil nyata menunjukkan set itu terlalu kecil untuk menilai.

## 7. Integrasi dan peluncuran bertahap
- Modul baru: `tools/memory_router.py` (inferensi, hanya numpy), `tools/make_router_data.py` (pembangkit), `tools/train_router.py` (pelatihan), berkas bobot kecil. Masuk `RUNTIME_TOOLS` dan paket rilis. Bila berkas bobot tidak ada, aplikasi memakai aturan lama.
- Alat uji kelulusan mendapat backend baru `router` agar classifier dinilai dengan skrip dan kriteria yang sama.
- **Mode bayangan (opsional, mati secara bawaan):** `route()` tetap memakai aturan; classifier dijalankan paralel dan perbedaannya dicatat ke berkas lokal `data/memory/router_bayangan.jsonl` hanya bila pemakai menyalakannya. **Pesan pemakai adalah data pribadi:** hanya di komputer itu, tidak dikirim ke mana pun, dan pemakai bisa menghapusnya. Hanya setelah tinjauan perbedaan (G5/G6) classifier menggantikan aturan.

## 8. Risiko
| Risiko | Mitigasi |
|---|---|
| Gaya penulis tunggal membuat angka terlalu bagus | Pesan tulisan Anda dan pesan nyata sebagai ujian independen; tidak melatih dari set uji |
| Classifier memakai jalan pintas (panjang kalimat, satu kata pemicu) | Seimbangkan panjang, jebakan ≥ 500 di train dan ≥ 120 di uji, periksa bobot fitur |
| Set uji kecil, angka tidak stabil | Set validasi besar, selang kepercayaan, laporkan jujur |
| Perubahan perilaku bagi pemakai | Mode bayangan, aturan sebagai pengaman, alat memverifikasi hasil |
| Pesan pemakai bersifat pribadi | Pencatatan opsional, lokal, tidak dikirim |

## 9. Yang akan dikerjakan setelah lampu hijau
1. `make_router_data.py` dan set validasi/jebakan/berisik, dengan pemeriksaan kebocoran dan sampel untuk Anda periksa.
2. `train_router.py` (pilihan A) lalu uji pada validasi; naik ke B bila perlu.
3. `memory_router.py`, kebijakan "mengusulkan, alat memverifikasi", dan backend `router` di uji kelulusan.
4. Tes otomatis, laporan terukur dengan selang kepercayaan, dan keputusan lulus/gagal.
5. Mode bayangan, lalu tinjauan perbedaan sebelum penggantian.

## 10. Keputusan yang diminta dari pemilik proyek
1. Bersedia menulis **±100 pesan bebas** (±15–20 menit) sebagai ujian gaya independen, dan memeriksa **150 sampel label**?
2. Setuju dengan pintu G2 sampai G6 (G1 sudah disetujui)?
3. Setuju dengan rancangan mode bayangan yang opsional dan lokal?
4. Mulai dari pilihan A (regresi logistik)?

## 11. Catatan pelaksanaan langkah 1 (data)
- **Jumlah (dibangkitkan dengan benih 42):** latih ±17.300 (cari 8.168, tolak 2.342, hitung 2.340, tanggal 1.996, urut 1.644, sapaan 801); validasi 2.100 (keluarga B); jebakan uji 150; berisik 591 (tiga varian bergangguan untuk tiap pesan uji resmi). Set validasi, jebakan, dan berisik **dibekukan**: pembangkit memuatnya apa adanya bila berkasnya ada, kecuali dengan `--refresh-evalsets`.
- **Kesalahan yang ketemu dan diperbaiki saat membangun data:** (1) ruang templat sapaan kecil sehingga hampir semua sapaan latih berisi salah ketik, diatasi dengan kuota salah ketik dan perluasan templat sapaan; (2) awalan ganda dan awalan sopan pada sapaan ("tolong selamat pagi") tidak wajar; (3) objek "tolak" tidak cocok dengan kata kerjanya ("buat rapat kemarin"); (4) 15% varian "berisik" ternyata tidak berubah dari aslinya karena bendera perubahan menyala tanpa efek. Semua dijaga tes di `tests/test_router_data.py`.
- **Keterbatasan yang tersisa:** panjang pesan antar kelas belum persis sama (median sapaan 3 kata, urut 7); sapaan memang pendek. Validasi memakai pesan lebih panjang dari latih (daftar angka yang panjang), yang sengaja menguji generalisasi tetapi juga bisa menurunkan skor. Seluruh data masih ditulis oleh satu penulis (saya); karena itu ujian independen (pesan tulisan Anda) tetap diperlukan.
- **Patokan aturan di kode pada set baru** (bukan classifier): lihat laporan pada pesan pelaksanaan; patokan ini menjadi pembanding untuk pintu G2 sampai G4.
