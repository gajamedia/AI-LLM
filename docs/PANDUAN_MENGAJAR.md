# Panduan Mengajar Memori

Aplikasi **Memori** memungkinkan pemilik komputer mengajari AI-nya fakta baru dengan kalimat biasa, lalu menanyakannya
kembali. Setiap jawaban disertai **catatan sumber**; jika tidak ada catatan yang meyakinkan, AI berkata
"belum tahu" dan tidak menebak.

## Cara kerjanya (singkat)

- **Model bahasa (100 juta parameter) tidak berubah.** Ia hanya bertugas *membaca* catatan dan menjawab.
- **Pengetahuan ada di berkas memori** (`data/memory/memory.db`, SQLite) milik Anda sendiri. Mengajar = menambah
  catatan; melupakan = menghapus catatan. Tidak ada pelatihan ulang, tidak butuh GPU, dan tidak ada data yang keluar dari komputer.
- Saat ditanya: sistem mencari beberapa catatan yang paling cocok, model membaca tiap catatan, lalu **pagar pengaman**
  hanya meloloskan jawaban yang benar-benar berupa potongan catatan tersebut.

## Memulai

```bash
python tools/memory_web.py --open        # antarmuka web di http://127.0.0.1:8765
python tools/memory_chat.py              # versi baris perintah
```

Pertama kali berjalan, model dimuat sekitar 10–20 detik. Jawaban berikutnya biasanya kurang dari 2 detik di CPU.
Opsi: `--checkpoint <berkas>` (model), `--db <berkas>` (lokasi memori), `--port <nomor>`.

## Aturan emas saat mengajar

Model ini kecil, jadi **kalimat yang sederhana jauh lebih akurat**.

1. **Satu fakta per catatan.** Pecah kalimat panjang.
2. **Sebut nama lengkap**, jangan "dia", "ia", "mereka".
3. **Tulis tanggal atau hari yang pasti**, bukan "kemarin", "besok", "sekarang" (akan basi).
4. **Subjek di awal, lalu keterangan.** Contoh: "Bu Rina tinggal di Jalan Mawar nomor 5."
5. **Sertakan satuan dan angka secara jelas:** "Harga satu kotak donat adalah 45 ribu rupiah."
6. **Perbarui dengan benar:** hapus catatan lama (atau tombol **Ubah**), lalu simpan yang baru.

| Kurang baik | Lebih baik |
|---|---|
| Dia pindah ke sana kemarin dan sekarang bekerja sebagai guru sambil mengurus kebun dan toko. | Bu Rina pindah ke Bandung pada 3 Maret 2026. <br> Bu Rina bekerja sebagai guru. <br> Bu Rina mengurus sebuah toko. |
| Rapatnya jadi besok. | Rapat tim produk dipindahkan ke hari Kamis pukul 09.30 di ruang Anggrek. |
| Harganya naik lagi. | Harga satu kotak donat naik menjadi 50 ribu rupiah mulai 1 April 2026. |

Antarmuka web memberi saran otomatis saat Anda mengetik (kata ganti di awal, kata waktu relatif, kalimat terlalu panjang).

## Cara bertanya

- Tanyakan dengan kalimat sendiri; tidak harus sama dengan kata pada catatan ("Apa pekerjaan Pak Joko?" untuk
  catatan "Pak Joko bekerja sebagai montir").
- **Sebut nama yang sama persis** dengan yang dicatat. Nama dan kata bilangan di pertanyaan harus ada di catatan.
- Satu pertanyaan, satu hal yang ditanyakan.

## Membaca jawaban

| Tampilan | Artinya |
|---|---|
| **Dari catatan** + sumber disorot | Jawaban dikutip dari catatan yang ditampilkan. Selalu periksa sumbernya. |
| **Ada konflik** | Beberapa catatan menjawab berbeda; yang dipakai catatan terbaru, yang lain ditampilkan. |
| **Belum tahu** | Tidak ada catatan yang meyakinkan. AI tidak menebak. Ajari jawabannya jika Anda mau. |
| Kotak kuning **"ada catatan lebih baru…"** | Ada catatan lebih baru yang mirip tetapi tidak bisa dibaca AI. Periksa sendiri; mungkin informasinya sudah diperbarui. |

**Ketelitian jawaban** (di bawah kolom pertanyaan):

- **Longgar** — lebih sering menjawab, lebih mungkin keliru.
- **Seimbang** (bawaan) — kompromi.
- **Ketat** — lebih sering "belum tahu", hampir tidak pernah mengarang. Pilih ini jika kesalahan lebih merugikan daripada tidak ada jawaban.

## Kemampuan dan batasnya (angka sebenarnya)

Angka berikut dari pengujian di repositori ini (`reports/eval_*.json`), memakai pengaturan **Seimbang**:

| Jenis catatan | Benar | Salah | "Belum tahu" | Mengarang pada soal yang tak punya jawaban |
|---|---|---|---|---|
| Fakta pendek terstruktur (memori 200 catatan, uji sintetis) | ~74% | 0% | ~26% | 0% |
| Kalimat bebas sederhana (tulisan tangan, 20 soal) | ~50% | ~5% | ~45% | 0% (8 soal) |
| Kalimat rumit, banyak fakta (tulisan tangan, 69 soal) | ~39% | ~14% | ~46% | ~7% (15 soal) |

Yang perlu dipahami:

- Ini **model kecil**. Jangan gunakan sebagai satu-satunya sumber untuk keputusan penting (kesehatan, hukum, keuangan).
- Ketika tidak yakin, ia **lebih sering memilih diam** daripada salah. Itu disengaja, tetapi artinya sebagian pertanyaan
  yang sebenarnya terjawab akan dijawab "belum tahu". Kalimat yang lebih sederhana mengurangi hal ini.
- Jumlah soal pengujian kecil (puluhan), jadi angka di atas adalah gambaran kasar, bukan jaminan.
- Pembaruan fakta tidak otomatis menggantikan catatan lama jika kalimatnya berbeda jauh. Hapus catatan lama.
- Model tidak "mengetahui" apa pun di luar catatan Anda secara andal; pengetahuan umumnya lemah dan sering keliru.

## Privasi dan cadangan

- Semua catatan tersimpan lokal di `data/memory/memory.db`. Server hanya mendengarkan `127.0.0.1` (komputer ini saja)
  dan tidak memakai autentikasi: **jangan membukanya ke jaringan atau internet**.
- **Cadangkan** dengan menyalin berkas `data/memory/memory.db` (tutup aplikasi dulu agar konsisten).
- Memori tiap pengguna terpisah: bagikan **model** (berkas `.pt`) ke banyak orang, tetapi tiap orang memakai berkas memorinya sendiri.
- Folder `data/` tidak masuk git, sehingga catatan pribadi tidak ikut terkirim ke GitHub.

## Pemecahan masalah

| Gejala | Penyebab / solusi |
|---|---|
| "Belum tahu" padahal sudah diajarkan | Tulis ulang catatan lebih sederhana; samakan nama pada pertanyaan; coba **Longgar**. |
| Jawaban keliru | Periksa sumber yang ditampilkan; sederhanakan catatan; gunakan **Ketat**; hapus catatan yang meragukan. |
| Jawaban lama masih muncul setelah memperbarui | Hapus catatan lama (nomor ditampilkan di sumber). |
| Halaman tidak terbuka | Pastikan server berjalan dan buka `http://127.0.0.1:8765/` (bukan alamat jaringan). |
| Model gagal dimuat | Periksa `--checkpoint`; berkas `.pt` harus berupa satu file (bukan folder). |
