# Panduan Instalasi dan Pemakaian: Memori

Panduan langkah demi langkah untuk memasang dan menjalankan **Memori** di komputer Anda. Tidak perlu kemampuan pemrograman.
Setelah terpasang, aplikasi berjalan **sepenuhnya offline**. Cara mengajar AI ada di `PANDUAN_MENGAJAR.md`.

## 1. Yang Anda butuhkan

| | Kebutuhan |
|---|---|
| Sistem operasi | **Windows 10/11** (sudah diuji). Linux dan macOS: skrip tersedia tetapi **belum diuji** (lihat bagian 9). |
| Python | **3.10 atau lebih baru** (gratis, cara memasang di bawah) |
| RAM | Aplikasi memakai ±1,0 GB (puncak 1,1 GB). Disarankan RAM total 4 GB atau lebih. |
| Ruang disk | ±1,2 GB (paket 388 MB + lingkungan Python ±0,7 GB) |
| Internet | Hanya untuk **pemasangan pertama** (±250 MB). Setelah itu tidak diperlukan. |
| Kartu grafis (GPU) | **Tidak perlu.** Berjalan di prosesor biasa. |

Kecepatan (diukur di komputer 8 inti): model dimuat dalam 5–10 detik, jawaban sekitar 0,5–2 detik. Komputer yang lebih lambat akan lebih lama.

## 2. Unduh dan periksa berkas (opsional tetapi disarankan)

1. Unduh `hai07-memori-v….zip` dari halaman Releases di GitHub.
2. Periksa keutuhan. Di PowerShell:
   ```powershell
   Get-FileHash .\hai07-memori-v1.1126.12.zip -Algorithm SHA256
   ```
   Bandingkan dengan nilai SHA-256 yang tertulis di catatan rilis. Jika berbeda, unduh ulang (unduhan terputus atau rusak).

## 3. Pasang Python (sekali saja)

1. Buka https://www.python.org/downloads/release/python-3120/ dan unduh **Windows installer (64-bit)** Python 3.12. Disarankan 3.12 karena
   aplikasi hanya diuji pada Python 3.10; versi yang jauh lebih baru bisa gagal saat memasang pustaka.
2. Jalankan pemasangnya dan **centang kotak "Add python.exe to PATH"** di layar pertama, lalu **Install Now**.
3. Cek: buka *Command Prompt* (ketik `cmd` di menu Start) dan ketik `python --version`. Harus muncul `Python 3.10` sampai `3.12`.

Jika perintah itu tidak dikenali, atau yang muncul adalah jendela Microsoft Store, pemasangan Python perlu diulang dengan kotak PATH dicentang
(Windows punya pintasan `python` palsu yang membuka Store; `jalankan.bat` akan memberi tahu bila itu yang terdeteksi).

## 4. Ekstrak paket

1. Klik kanan zip, pilih **Extract All…**, dan simpan di folder yang Anda miliki, misalnya `C:\Memori` atau `Documents`.
   Jangan jalankan dari dalam zip, dan hindari `C:\Program Files` (butuh izin khusus untuk menulis).
2. Hasilnya folder `hai07-memori-v1.1126.12` berisi `jalankan.bat`, `model`, `tools`, dll.

## 5. Jalankan pertama kali

1. Klik ganda **`jalankan.bat`**.
2. Pertama kali, skrip menyiapkan lingkungan Python dan mengunduh pustaka (±250 MB). Pada pengujian kami, **sekitar 6–7 menit**; bergantung kecepatan internet. Biarkan jendela hitam terbuka sampai muncul tulisan seperti:
   ```
   [web] model siap (… dtk).
   [web] buka http://127.0.0.1:8765/
   ```
3. Peramban terbuka otomatis. Jika tidak, buka alamat `http://127.0.0.1:8765` sendiri.

Jika Windows menampilkan peringatan keamanan untuk berkas `.bat` (misalnya "Windows protected your PC"), itu karena berkas berasal dari internet.
Lanjutkan hanya jika Anda mempercayai sumber unduhan, dan hanya jika checksum di bagian 2 cocok.

## 6. Memakai aplikasi

1. **Ajari** (kolom kanan atas): tulis satu fakta, misalnya `Bu Rina tinggal di Jalan Mawar nomor 5.`, lalu **Simpan ke memori**.
2. **Tanya** (kolom kiri): ketik `Di mana Bu Rina tinggal?` lalu Enter. Jawaban muncul beserta **sumbernya**.
3. Jika AI menjawab "Saya belum tahu", itu normal. Ia memilih diam daripada menebak. Coba tulis ulang catatan lebih sederhana, atau pilih **Longgar** di bawah kolom pertanyaan.
4. **Dokumen:** di bawah kotak Ajari AI ada "Impor dokumen" (berkas `.txt` atau `.md`). AI lalu menampilkan **bagian dokumen yang paling mirip** beserta sumbernya; jawaban AI hanya dugaan dan sering keliru, jadi baca sendiri bagian yang tampil.
5. Ikon **?** di pojok kanan atas membuka tips singkat mengajar. Panduan lengkap: `PANDUAN_MENGAJAR.md`.

Tips utama: **satu fakta per kalimat, sebut nama lengkap, tulis tanggal yang pasti, dan selalu periksa sumber jawaban.**

## 7. Menjalankan lagi, menghentikan, mengubah port

- **Menjalankan lagi:** klik ganda `jalankan.bat`. Berikutnya hanya butuh 10–20 detik (tanpa unduhan).
- **Menghentikan:** tekan `Ctrl+C` di jendela hitam atau tutup jendelanya. Catatan Anda tetap tersimpan.
- **Port 8765 sudah dipakai** (misalnya Memori sudah berjalan di jendela lain, atau program lain): aplikasi langsung berhenti dengan pesan
  "Port 8765 sudah dipakai program lain". Tutup yang lama, atau jalankan lewat Command Prompt di folder paket:
  `jalankan.bat --port 8800`, lalu buka `http://127.0.0.1:8800`.
  Jangan menjalankan dua salinan Memori sekaligus dengan memori yang sama.

## 8. Cadangan, pembaruan, dan mencopot

- **Catatan Anda** ada di satu berkas: `data\memory\memory.db`. **Cadangkan berkas itu** secara berkala (tutup aplikasi dulu agar konsisten).
- **Pindah ke komputer lain / memulihkan:** pasang paket di komputer baru, lalu salin `memory.db` ke `data\memory\`.
- **Memperbarui ke versi baru:** ekstrak versi baru ke **folder baru**, lalu salin `data\memory\memory.db` dari folder lama ke folder baru. Jangan menimpa folder lama sebelum catatan Anda tersalin.
- **Mencopot:** tutup aplikasi dan hapus seluruh folder paket (termasuk `.venv`). Tidak ada yang terpasang di luar folder itu, kecuali Python sendiri. Jangan lupa cadangkan `memory.db` jika masih ingin menyimpan catatan.
- **Banyak pengguna:** model bisa dibagikan, tetapi tiap pengguna sebaiknya punya folder (dan `memory.db`) sendiri agar catatannya tidak tercampur.

## 9. Linux dan macOS (belum teruji)

Skrip `jalankan.sh` disertakan tetapi **belum pernah dijalankan** karena pembuatnya hanya menguji di Windows. Anggap ini percobaan dan laporkan jika gagal.

```bash
# Debian/Ubuntu: pastikan Python dan venv tersedia
sudo apt install python3 python3-venv
cd hai07-memori-v1.1126.12
bash jalankan.sh
```

Di macOS pasang Python 3.10+ dari python.org atau Homebrew, lalu jalankan `bash jalankan.sh`. Di Linux, skrip memasang versi torch untuk CPU saja (lebih kecil).

## 10. Pemecahan masalah

| Gejala | Penyebab dan solusi |
|---|---|
| "Python 3.10 atau lebih baru belum terpasang" | Pasang Python (bagian 3) dan **centang Add python.exe to PATH**, lalu klik ganda `jalankan.bat` lagi. |
| "Gagal menyiapkan lingkungan" | Koneksi internet terputus atau diblokir (proxy, VPN, antivirus), atau versi Python tidak cocok (pasang Python 3.12). Hapus folder `.venv` di dalam paket, lalu jalankan lagi. |
| Pemasangan sangat lama | Normal pada internet lambat (±250 MB). Biarkan sampai selesai. |
| Jendela hitam langsung tertutup | Jalankan dari Command Prompt agar pesan galat terbaca: buka `cmd`, `cd` ke folder paket, ketik `jalankan.bat`. |
| Model gagal dimuat | Ukuran `model\hai07-100m-v….pt` harus ±404 MB. Jika lebih kecil, unduhan rusak: unduh ulang dan periksa checksum. |
| Halaman tidak terbuka | Pastikan jendela hitam masih berjalan dan buka `http://127.0.0.1:8765/` (bukan alamat jaringan). |
| "Port 8765 sudah dipakai program lain" | Memori mungkin sudah berjalan di jendela lain: tutup yang lama, atau pakai port lain (bagian 7). |
| Meminta AI menghitung, mengurutkan, atau menulis | AI hanya membaca catatan. Hanya alat kecil yang tersedia: urutkan angka, hitung (+ - x / ^), hari dan selisih tanggal. Lihat `PANDUAN_MENGAJAR.md`. |
| AI sering menjawab "belum tahu" | Tulis catatan lebih sederhana, samakan nama dengan di catatan, atau pilih **Longgar**. Ini batas model kecil, bukan kerusakan. |
| Komputer terasa berat saat menjawab | Tutup program lain; model memakai ±1 GB RAM dan prosesor saat menjawab. |
| Catatan hilang | Periksa `data\memory\memory.db` di folder yang Anda jalankan. Memori ikut folder paketnya. Jika Anda membuka folder paket yang berbeda, memorinya juga berbeda. |

Jika masalah tetap ada, laporkan di https://github.com/gajamedia/AI-LLM/issues dengan menyertakan sistem operasi, versi Python (`python --version`), dan teks pesan galat dari jendela hitam.

## 11. Privasi dan keamanan

- Semua catatan dan model berada di komputer Anda. **Tidak ada data yang dikirim ke internet** saat Anda memakai aplikasi (internet hanya dipakai saat pemasangan awal).
- Server hanya menerima koneksi dari komputer ini (`127.0.0.1`) dan **tidak memakai kata sandi**. Jangan mengubahnya agar bisa diakses lewat jaringan atau internet.
- AI ini kecil dan bisa keliru. **Jangan dipakai sendirian untuk keputusan penting** (kesehatan, hukum, keuangan). Selalu periksa sumber yang ditampilkan.
- Lisensi dan sumber data: lihat `LISENSI.txt` dan `NOTICE.md` (bobot model untuk penggunaan **non-komersial**).
