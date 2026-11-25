# Uji pengguna Notis (hai07) v1 (jalur C: gratis, non-komersial)

Tujuan: mengukur apakah aplikasi memori lokal berguna bagi orang selain pengembang, dengan catatan mereka sendiri.
Tanggal mulai: 2026-11-25. Evaluasi keputusan: 2027-01-05 (6 minggu).

## Peserta
- 8 sampai 10 orang di luar pengembang; minimal 3 sampai 4 orang non-teknis.
- Tidak ada pembayaran dan tidak ada janji produk komersial.

## Lembar persetujuan (dibaca dan disetujui peserta)

Saya memahami bahwa:
1. Ini versi uji **non-komersial** dari aplikasi Notis (hai07). Model dan aplikasi diberikan apa adanya, tanpa jaminan.
2. Model kecil dan bisa salah atau menjawab "belum tahu". Saya tidak akan menyimpan hal penting (kata sandi, data keuangan, data kesehatan) sebagai satu-satunya salinan.
3. Catatan saya disimpan **hanya di perangkat saya**. Pengembang tidak menerima catatan saya kecuali saya sendiri mengirimkannya.
4. Pencatatan penggunaan (opsional, pilih satu):
   - [ ] TIDAK: tidak ada log yang dikirim.
   - [ ] Log ringan: hanya jenis pesan dan jumlah kata, tanpa isi teks.
   - [ ] Contoh pesan: saya memilih sendiri beberapa pesan (boleh diedit/disamarkan) untuk dibagikan dan diberi label.
5. Saya boleh berhenti kapan saja dan meminta data yang saya kirim dihapus.

Nama / inisial: ______________  Tanggal: ____________  Tanda tangan/persetujuan tertulis: ____________

## Menyiapkan pencatatan log (dilakukan pewawancara, bukan peserta)
Hanya untuk peserta yang memilih "log ringan" atau "contoh pesan". Salin `memori.env.example` menjadi `memori.env` di folder paket peserta, lalu:
- Pastikan `MEMORI_ROUTER=shadow` (classifier jalan paralel, jawaban tidak berubah).
- **Buka tanda `#` pada `MEMORI_ROUTER_LOG=data/memory/router_log.jsonl`.** Di berkas contoh baris ini dikomentari, jadi tanpa langkah ini tidak ada yang tercatat.
- Untuk opsi "contoh pesan" saja, buka juga `MEMORI_ROUTER_LOG_TEXT=1` (menyimpan teks pesan, hanya di komputer peserta).
- Log hanya mencatat jenis pesan untuk router. **Benar atau salahnya jawaban tidak tercatat**, jadi formulir di bawah tetap wajib diisi.
- Memberi label pada pesan: `python tools/router_log_tools.py to-md ...` lalu `from-md ...` (lihat komentar di awal berkas itu).

## Panduan peserta (satu halaman)
- Bagikan `docs/PANDUAN_PESERTA.md` (ikut dalam paket rilis berikutnya); instalasi lengkap ada di `docs/PANDUAN_INSTALASI.md`. Catat jika ada langkah yang membingungkan.
- Selama 7 hari, simpan catatan sungguhan dengan `ingat: ...` dan bertanya seperti biasa.
- Jangan menyesuaikan gaya tulisan agar "dimengerti" aplikasi; tulis natural.
- Setiap kali jawaban salah atau terasa aneh, catat di formulir di bawah.

## Formulir pencatatan (satu baris per pertanyaan; boleh dicatat peserta atau pewawancara)

| Hari | Pertanyaan (boleh disamarkan) | Hasil: Benar / Salah / Belum tahu | Seharusnya "belum tahu"? (ya/tidak) | Catatan |
|---|---|---|---|---|

Pertanyaan akhir (hari ke-7):
1. Apakah Anda memakai aplikasi ini pada hari ke-3? Hari ke-7? (ya/tidak)
2. Hal apa yang paling berguna? Paling mengganggu?
3. Apakah ada jawaban salah yang Anda percayai sebelum memeriksanya?
4. Apakah instalasi selesai tanpa bantuan? Di langkah mana macet?
5. Akankah Anda terus memakainya? Berapa nilainya bagi Anda (0 sampai 10)?

## Metrik keputusan (dihitung dari formulir)
- Terjawab wajar = benar, atau "belum tahu" pada pertanyaan yang memang tidak ada di catatan. Target >= 70%.
- Salah-mengarang = jawaban salah yang dinyatakan yakin. Target <= 5%.
- Retensi = peserta yang masih memakai di hari ke-7. Target >= 50%.
- Ukuran sampel kecil: selisih beberapa poin bukan bukti. Laporkan angka mentah (X dari N).

## Aturan keputusan pada 2027-01-05
- Ketiga metrik tercapai: lanjut ke jalur B (ulang SFT tanpa data bermasalah) dan pertimbangkan rilis komersial.
- Sebagian tercapai: perbaiki (fase 3) lalu uji ulang dengan peserta baru.
- Tidak tercapai: jangan investasi lisensi komersial; tinjau ulang posisi produk.

## Penanganan data peserta
- Contoh pesan yang dibagikan disimpan lokal, tidak masuk git, dan dihapus atas permintaan.
- Hanya pesan yang diberi persetujuan yang boleh dipakai untuk label classifier router.
