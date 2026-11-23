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

## Yang tidak bisa dilakukan: ini bukan asisten umum

AI hanya **menyalin potongan dari catatan Anda**. Ia tidak bisa menghitung, mengurutkan, menalar, menerjemahkan, merangkum, atau menulis.
Catatan seperti "urutan angka satuan: 0, 1, 2, …, 9" hanya berisi daftar, bukan cara mengurutkan angka baru, dan prosedur tidak bisa
diajarkan lewat catatan fakta. Model sekecil ini juga tidak dilatih bernalar (hasil uji: diminta mengurutkan "5, 4, 3, 9" ia mengeluarkan
"5, 9, 9, 10", dan diminta menjumlahkan "3 apel dan 4 jeruk" ia hanya mengutip "3").

Karena itu, aplikasi punya **alat kecil yang dikerjakan program, bukan model**. Hasilnya selalu benar untuk pola yang dikenal dan diberi label
"Alat" (bukan "Dari catatan"):

| Ketik | Hasil |
|---|---|
| `urutkan 5, 4, 3, 9` atau `urutkan deret angka ini (5-4-3-9)` | `3, 4, 5, 9` (tambah `dari besar ke kecil` untuk menurun) |
| `berapa 3 + 4`, `hitung 12 x 5`, `100 dibagi 4`, `2 pangkat 10` | `7`, `60`, `25`, `1024` |
| `hari apa 17 Agustus 2026` | `Senin` |
| `berapa hari antara 1 Januari 2026 dan 1 Maret 2026` | `59 hari` |

Catatan alat: angka boleh ditulis dengan kata (`berapa satu ditambah satu`, `urutkan tiga, satu, dua`, `dua puluh lima kali dua`), angka desimal
memakai koma (`3,5`), titik ribuan boleh (`1.000`), hanya tanggal lengkap dengan nama bulan yang dikenali, dan alat hanya mengenali pola di atas
(bukan bahasa bebas). Permintaan lain ("terjemahkan…", "tuliskan…", "ringkaskan…") dijawab dengan penjelasan bahwa aplikasi tidak bisa
mengerjakannya. Pertanyaan biasa tentang catatan tidak terpengaruh (mis. "Siapa yang lahir tanggal 12-03-1990?" tidak dianggap pengurangan).
Tabel atau rumus (mis. `1+1=2 1+2=3 …`) tidak bisa dipelajari dari catatan; ketik saja perhitungannya.

## Aturan: jawaban tetap untuk kalimat tertentu

Anda bisa mengajarkan **balasan tetap**: bila Anda mengetik kalimat tertentu, AI menjawab dengan balasan yang Anda tentukan. Tulis catatan
berbentuk `jika … "pemicu" … jawab … "balasan"`:

```
ingat: jika ada yang menyapa "halo" jawab saja "halo"
ingat: kalau ditanya "siapa kamu" jawab "Saya memori pribadi Andi."
```

Setelah itu `halo` dijawab `halo`, dan `siapa kamu` dijawab `Saya memori pribadi Andi.`. Jawabannya diberi label **Aturan**, lengkap dengan catatan
sumbernya. Aturan dijalankan **program**, bukan model, jadi sifatnya kaku:

- **Pemicu harus persis sama** dengan yang Anda ketik (huruf besar/kecil dan tanda baca diabaikan). `Halo apa kabar` **tidak** memicu aturan `"halo"`.
- Pemicu dan balasan **wajib diberi tanda kutip** (`"…"`). Tanpa tanda kutip, catatan dianggap catatan biasa.
- Aturan terbaru menang bila ada beberapa aturan dengan pemicu sama. Hapus catatan aturan untuk menonaktifkannya.
- Aturan mengalahkan sapaan bawaan, dan hanya menjawab teks tetap: ia tidak bisa menalar atau menyusun jawaban baru.

## Menempel beberapa baris sekaligus

Di web, kotak "Ajari AI" menerima beberapa baris sekaligus (maksimal 60 baris dan 6.000 karakter): **tiap baris disimpan sebagai satu catatan terpisah**.
Penanda daftar di awal baris (`-`, `*`, bullet, `1.`) dibuang; baris kosong dan baris berisi satu kata (mis. judul "Catatan:") dilewati. Sebelum
menyimpan, aplikasi menampilkan jumlah catatan yang akan dibuat dan saran per baris; bila ada baris di atas 600 karakter, tidak ada yang disimpan.
Pengecualian: daftar kamus "1 = siji 2 = loro ..." tetap satu catatan.

Ini penting karena satu catatan berisi banyak baris dilebur menjadi satu kalimat panjang dan model sulit memisahkan nilainya. Uji dengan 5 pertanyaan
tentang "Koperasi Maju": satu catatan banyak baris hanya **2 dari 5 benar** (dengan jawaban terpotong seperti "Pak Slamet Jumlah anggota" dan "24"),
sedangkan satu baris per catatan dengan kalimat lengkap **5 dari 5 benar**. Tulis tiap baris sebagai kalimat lengkap yang mengulang nama, mis.
`Ketua koperasi Maju adalah Bu Ratna.` (jumlah soal uji kecil; ini gambaran, bukan jaminan).

## Jenis pertanyaan yang bisa dan belum bisa dijawab

Model dilatih mengutip jawaban untuk **jenis fakta tertentu**: kota lahir, tahun lahir, profesi, makanan, hewan, lantai, mobil, tenggat,
harga, ketua, telepon, dan alamat. Jam, ruang, tanggal, orang, dan tempat yang sederhana umumnya juga terbaca karena mirip jenis itu.
Untuk jenis lain, model sering tidak bisa mengutip jawabannya walau catatannya jelas, dan aplikasi akan menjawab "belum tahu" (lengkap
dengan jawaban mentah model yang ditolak dan alasannya). Itu keterbatasan model, bukan kesalahan catatan Anda.

| Bisa (umumnya) | Belum bisa (diukur) |
|---|---|
| "Jam berapa rapat dimulai di ruang Mawar?" | "Siapa yang rapat di ruang Anggrek?" bila jawabannya nama orang atau peran |
| "Rapat tim finance ada di ruang apa?" | Pertanyaan "siapa ..." dan pola lain di luar "kategori mana/apa" |
| "Di mana Bu Rina tinggal?", "Siapa ketua koperasi Maju?" | Jenis fakta di luar daftar di atas (nama divisi, peran, dan sejenisnya) |

Pengujian pada model saja: dari 35 percobaan (5 gaya catatan × 7 susunan pertanyaan) untuk mengeluarkan nama tim, **tidak ada satu pun** jawaban
bersih; model mengulang kata dari pertanyaan, menjawab jam, atau menolak. Karena itu aplikasi punya **cadangan** untuk satu pola: pertanyaan
berbentuk "**KATEGORI mana/apa** ..." (mis. "Tim mana yang rapat di ruang Anggrek?"). Hanya bila model tidak menjawab, program mengambil kata PERSIS
yang mengikuti kategori di catatan ("Rapat tim produk ... di ruang Anggrek" memberi "produk"). Pada uji aplikasi utuh, pertanyaan nama tim naik
dari 0 menjadi 3 dari 3 (jumlah soal sangat kecil, ini gambaran, bukan jaminan). Cadangan ini berhenti dan tetap menjawab "belum tahu" bila ruang
atau entitas yang ditanya tidak ada di catatan, ada dua jawaban berbeda, atau tidak ada kata pembeda. Pola lain belum ditangani: **tanyakan hal yang
bisa dijawab** (jam, ruang, tanggal), atau tanyakan secara umum ("rapat ada di ruang apa") dan pilih dari **pertanyaan balik** aplikasi.
Menambah kemampuan jenis fakta baru memerlukan pelatihan ulang model.

## Pertanyaan lanjutan ("dan istrinya?")

Aplikasi mengingat beberapa pertanyaan terakhir dalam sesi dan memahami pertanyaan lanjutan yang pendek: `Apa pekerjaannya?`,
`Di mana dia tinggal?`, `kalau Pak Joko?`, `dan istrinya?`. Penulisan ulang dilakukan **program, bukan model**, dan selalu ditampilkan
("Dibaca sebagai: ..."), jadi Anda bisa memeriksanya. Aturannya sempit dan hanya berlaku bila tandanya jelas: ada kata ganti
(dia/ia/beliau/mereka), kata berakhiran -nya, atau awalan penyambung (dan/kalau/lalu/bagaimana dengan), dan pertanyaan sebelumnya memuat nama
(kata berhuruf kapital). Pertanyaan yang sudah lengkap tidak diubah. Bila rujukannya meleset, tulis nama lengkap. Riwayat hanya di sesi
berjalan (CLI: sampai ditutup; web: sampai halaman dimuat ulang) dan tidak disimpan. Jawabannya tetap melewati penjaga yang sama.

## Mengolah angka di catatan

Alat bisa mengambil angka dari **satu catatan yang Anda sebut jelas**, lalu mengolahnya: `urutkan angka di catatan 3`,
`jumlahkan angka di catatan 3`, `rata-rata angka di catatan 3`, `angka terbesar di catatan terakhir` (juga terkecil, dan "dari terbesar").
Contoh catatan: `Skor ujian: 78, 95, 60, 88.` Jawabannya selalu disertai angka yang dibaca dari catatan itu, jadi mudah diperiksa.
Hati-hati: **tanggal, jam, dan nomor telepon di catatan ikut terbaca sebagai angka**. Angka dari semua catatan sengaja tidak digabung
karena akan tercampur; sebutkan nomor catatannya (lihat nomor di daftar Catatan). Yang dibaca hanya angka berdigit, bukan kata bilangan,
dan hanya catatan yang Anda ajarkan (bukan potongan dokumen).

## Kamus: mengajari alat hitung/urut kata baru

Alat hitung dan pengurut membaca kata bilangan Indonesia bawaan (satu ... sembilan, belas, puluh, ratus, ribu, juta, miliar). Anda bisa
**menambah kata sendiri** (mis. bahasa daerah) lewat catatan berformat tetap, tanda kutip wajib:

| Catatan | Artinya |
|---|---|
| `kata "siji" berarti 1` | kata bilangan 0-9 (juga boleh "artinya", "sama dengan", "adalah") |
| `pengali "ewu" berarti 1000` | pengali kelipatan sepuluh, 1000 sampai 1.000.000.000.000 |
| `operator "campur" berarti +` | operator: `+ - * / ^` (atau tambah, kurang, kali, bagi, pangkat) |

Setelah itu: `berapa siji campur loro`, `urutkan telu, siji, loro`, `berapa lima ewu kali dua`. Aplikasi memberi tahu bila catatan dikenali
atau ditolak. Aturan keamanan: kata harus 2-20 huruf a-z tanpa spasi; kata yang sudah dipakai program (kata tanya, sapaan, perintah,
bulan/hari, bilangan/operator bawaan) ditolak; maksimal 200 entri; bila satu kata didefinisikan dua kali, **catatan terbaru menang**;
melupakan catatannya menghapus entrinya. Kosakata bawaan (satu...sembilan, ribu/juta/miliar, ditambah/kali/dibagi, dst.) dibaca dari berkas data `tools/kamus_bawaan.txt`, bukan dari kode; berkas itu boleh disunting, tetapi kata bawaan tidak bisa ditimpa catatan. Ini hanya perluasan kosakata: **cara menghitungnya tetap kode program**, dan kalimat biasa
yang kebetulan memuat kata kamus tidak dialihkan ke alat kecuali ada kata perintah/operator hitung.

## Belajar dari dokumen (impor berkas)

Anda bisa mengimpor dokumen **.txt atau .md** (di web: kotak "Impor dokumen" di bawah panel Ajari AI, boleh diseret; di baris perintah:
`impor <berkas>`, `dokumen`, `hapus dokumen <nomor>`). Dokumen dipecah per paragraf (maksimal ±500 karakter, judul Markdown ikut sebagai
awalan, butir daftar dipisah titik koma, blok kode dibuang) dan disimpan terpisah dari catatan Anda: dokumen tidak memenuhi daftar
Catatan, dan dihapus per dokumen. Mengimpor berkas dengan **nama yang sama menggantikan** versi lama. Hanya teks lokal; tidak ada yang dikirim keluar.

**Cara kerjanya dan batasnya, jujur:** AI **tidak memahami** dokumen. Ia mencari bagian yang paling mirip dengan pertanyaan Anda
(pencarian kata kunci) dan **menampilkan bagian itu beserta sumbernya** (nama berkas dan nomor bagian, kata kunci disorot). Jawaban AI hanya
**"dugaan"** yang diberi peringatan, karena pada paragraf yang memuat banyak fakta ia sering mengutip fakta yang salah.

| Uji | Hasil |
|---|---|
| 100 pertanyaan dari paragraf Wikipedia (±95 bagian) | Bagian yang memuat jawaban masuk 3 teratas pada **94%** soal; jawaban AI benar **13%**, salah 7%, "belum tahu" 80% |
| Dokumen realistis kecil (7 bagian, 12 pertanyaan) | Bagian yang benar tampil pada **12 dari 12** soal (peringkat 1); dari 8 kali AI berani menjawab, hanya **3 benar** dan 5 salah (mis. ketua dijawab nama sekretaris) |

Jadi: **gunakan fitur ini sebagai pencari lokal untuk dokumen Anda, dan baca sendiri bagian yang ditampilkan.** Jangan mengandalkan dugaan AI.
Dokumen kecil seperti uji kedua mudah dicari; dokumen panjang dengan banyak paragraf mirip akan menurunkan ketepatan pencarian (belum diukur).

Batas lain: PDF dan DOCX belum didukung (salin teksnya ke .txt; PDF hasil pindai tidak terbaca); tabel, gambar, dan blok kode di dalam dokumen
tidak ikut; pencarian memakai kata kunci sehingga sinonim dan parafrase jauh bisa meleset; berkas maksimal ±3 juta karakter (≤5.000 bagian).
Bila catatan Anda dan teks dokumen sama-sama menjawab, **catatan Anda didahulukan**. Berkas bukan UTF-8 dicoba dibaca sebagai Windows-1252;
bila teks tampak rusak, simpan ulang berkas sebagai UTF-8.

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
