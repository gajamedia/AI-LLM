"""
Pembangkit data pemilih alat (tahap 1): pesan berlabel untuk 6 kelas (urut, hitung, tanggal, cari, sapaan, tolak).
Rencana dan alasan rancangan: docs/RENCANA_TAHAP1_PEMILIH_ALAT.md. Semua data buatan sendiri (bebas masalah lisensi).

Prinsip:
  - Dua KELUARGA templat terpisah per kelas: A untuk melatih, B untuk validasi (struktur dan sebagian kata kerja tidak ada di A),
    sehingga validasi mengukur generalisasi ke templat tak terlihat, bukan hafalan templat.
  - 'cari' memuat banyak pesan JEBAKAN (pertanyaan biasa yang memuat kata pemicu alat: susun, tulis, total, jumlah, hari apa ...),
    keluarga A untuk latih dan keluarga B (set terpisah) untuk uji.
  - Kontrol kebocoran: pesan latih yang identik atau hampir sama (3-gram karakter, Jaccard >= 0,8) dengan pesan di set uji dibuang.
    Templat memang boleh tumpang tindih (itu yang diukur set validasi); yang dilarang adalah pesan yang nyaris sama.
  - Penyeimbangan panjang pesan antar kelas supaya classifier tidak memakai panjang kalimat sebagai jalan pintas.

  python tools/make_router_data.py                 # menulis data/router/ (latih, ringkasan, sampel ulasan) dan evalsets/router_*.jsonl
Keluaran evalsets (dibekukan setelah dibuat; tracked di git): router_validasi.jsonl, router_jebakan.jsonl, router_berisik.jsonl.
"""
import argparse
import collections
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_gen as G  # noqa: E402
import memory_tools as T  # noqa: E402

LABELS = ["urut", "hitung", "tanggal", "cari", "sapaan", "tolak"]
MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

NAMES = ["Bu Rina", "Pak Joko", "Dimas", "Bu Wati", "Pak Budi", "Sari", "Dokter Lina", "Pak Anto", "Bu Ratna", "Pak Slamet", "Mbak Tini", "Mas Doni", "Bu Marni",
         "Pak Hasan", "Ibu Yuni", "Pak Lurah", "Bu Dewi", "Pak Agus", "Kak Intan", "Pak Heru", "Bu Mawar", "Mas Bagas", "Mbak Citra", "Pak Dharma", "Bu Kartika", "Pak Rudi",
         "Bu Susi", "Pak Tono", "Mbak Wulan", "Pak Faisal", "Ibu Ningsih", "Pak Gunawan", "Bu Fitri", "Mas Arif", "Pak Yusuf", "Bu Lastri", "Mbak Dina", "Pak Eko", "Bu Hana", "Pak Bambang"]
ORGS = ["koperasi Maju", "RT 05", "tim produk", "tim finance", "panitia lomba", "klub catur", "kelas 6B", "bengkel Sentosa", "toko roti Cahaya", "posyandu Melati",
        "karang taruna", "perpustakaan desa", "warung Bu Ani", "sekolah dasar 3", "kantor desa", "tim humas", "panitia bazar", "kelompok tani"]
EVENTS = ["rapat evaluasi", "rapat RT", "ujian akhir", "pertemuan wali murid", "latihan paduan suara", "kerja bakti", "arisan", "pelatihan kasir", "lomba 17 Agustus",
          "bazar amal", "senam pagi", "rapat anggaran", "kunjungan dokter", "pengajian", "festival layang-layang", "rapat panitia"]
ITEMS = ["sepeda motor", "mesin kopi", "laptop kantor", "kunci gudang", "proyektor", "printer", "lemari arsip", "sepeda", "mobil dinas", "kamera", "meja lipat", "genset"]
OBJ_DOC = ["jadwal rapat", "laporan bulanan", "daftar peserta", "surat undangan", "anggaran tahunan", "notulen rapat", "daftar piket", "proposal kegiatan", "jadwal ronda",
           "laporan keuangan", "daftar hadir", "rencana kerja", "jadwal kuliah", "daftar belanja", "agenda pertemuan"]
WORDS = ["sinergi", "inflasi", "fotosintesis", "koperasi", "demokrasi", "gravitasi", "ekosistem", "anggaran", "notulen", "kuorum", "hipotesis", "erosi", "relevansi", "efisiensi"]
TOPICS = ["hujan", "ibu", "persahabatan", "kucing", "pahlawan", "laut", "sekolah", "kopi", "kampung halaman", "musim kemarau", "pasar tradisional", "sepak bola", "bulan purnama",
          "perjalanan kereta", "guru", "petani", "kemerdekaan", "jalan raya", "tukang bakso", "malam takbiran"]
CTX_NUM = ["angka", "bilangan", "deret", "nilai", "skor", "data", "daftar angka", "deretan angka"]

BUCKETS = [(1, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 10), (11, 99)]
BUCKET_WEIGHT = [0.08, 0.14, 0.16, 0.16, 0.14, 0.12, 0.09, 0.07, 0.04]     # sasaran panjang yang sama untuk semua kelas


# ---------- penyusun slot ----------
def _num_text(rng, allow_words=True):
    r = rng.random()
    n = rng.randint(1, 99) if r < 0.5 else rng.randint(100, 999) if r < 0.75 else rng.randint(1000, 99999) if r < 0.9 else None
    if n is None:
        return f"{rng.randint(0, 99)},{rng.randint(1, 9)}" if rng.random() < 0.6 else f"{rng.randint(0, 99)}.{rng.randint(1, 9)}"
    if allow_words and n <= 999 and rng.random() < 0.18:
        return T.spell_number(n)
    if n >= 1000 and rng.random() < 0.5:
        return f"{n:,}".replace(",", ".")
    return str(n)


def _nums(rng, k=None, joiner=None):
    k = k or rng.randint(3, 6)
    items = [_num_text(rng) for _ in range(k)]
    j = joiner or rng.choice([", ", " ", ", ", " - ", " dan ", "; "])
    if j == " dan ":
        return ", ".join(items[:-1]) + " dan " + items[-1]
    return j.join(items)


def _date(rng):
    y, m = rng.randint(1990, 2035), rng.randint(1, 12)
    d = rng.randint(1, 28)
    style = rng.random()
    if style < 0.7:
        return f"{d} {MONTHS[m - 1]} {y}"
    if style < 0.82:
        return f"{d} {MONTH_ABBR[m - 1]} {y}"
    if style < 0.92:
        return f"tanggal {d} {MONTHS[m - 1]} {y}"
    return f"{d}/{m}/{y}"


def _op(rng):
    return rng.choice(["ditambah", "dikurangi", "dikali", "dibagi", "plus", "minus", "kali", "pangkat", "dikalikan", "ditambahkan"])


def _sym(rng):
    return rng.choice(["+", "-", "x", "*", "/", "^", "+", "x"])


def _pick(rng, xs):
    return rng.choice(xs)


# ---------- keluarga templat: (nama, fungsi) ----------
def fam_urut_A():
    V = ["urutkan", "susun", "sortir", "rapikan", "sort", "urutin", "mengurutkan", "susunlah", "tolong urutkan", "mohon disusun"]
    DIR = ["", "", "dari kecil ke besar", "dari yang terkecil", "menaik", "naik", "dari terbesar", "dari besar ke kecil", "menurun", "turun", "ascending", "descending"]
    PAS = ["tolong diurutkan", "disusun", "dirapikan", "mohon diurutkan", "diurutkan ya"]
    return [
        ("urut_A1", lambda r: f"{_pick(r, V)} {_pick(r, CTX_NUM)} {_nums(r)} {_pick(r, DIR)}".strip()),
        ("urut_A2", lambda r: f"{_pick(r, V)} {_nums(r)} {_pick(r, DIR)}".strip()),
        ("urut_A3", lambda r: f"{_pick(r, CTX_NUM)} {_nums(r)} {_pick(r, PAS)} {_pick(r, DIR)}".strip()),
        ("urut_A4", lambda r: f"{_pick(r, V)} {_pick(r, CTX_NUM)} di catatan {r.choice(['terakhir', str(r.randint(1, 30)), '#' + str(r.randint(1, 30))])} {_pick(r, DIR)}".strip()),
        ("urut_A5", lambda r: f"bisakah {_pick(r, V[:6])} {_nums(r)}?"),
        ("urut_A6", lambda r: f"{_pick(r, V)} dari {_pick(r, ['yang kecil', 'yang terkecil', 'yang besar', 'terbesar'])}: {_nums(r)}"),
    ]


def fam_urut_B():
    VB = ["tata", "runtutkan", "atur urutan", "bikin berurutan", "jadikan berurutan", "tolong dibuat berurutan", "naikkan urutannya"]
    return [
        ("urut_B1", lambda r: f"mana yang paling kecil sampai paling besar dari {_nums(r)}"),
        ("urut_B2", lambda r: f"{_nums(r)} {_pick(r, VB)} {_pick(r, ['dari kecil', 'dari besar', 'menaik', 'menurun'])}"),
        ("urut_B3", lambda r: f"buat daftar berurutan dari {_nums(r)}"),
        ("urut_B4", lambda r: f"{_pick(r, VB)} deretan ini: {_nums(r)}"),
        ("urut_B5", lambda r: f"tolong {_pick(r, ['susunkan', 'rapikan', 'urutkan'])} jadi {_pick(r, ['naik', 'turun'])}: {_nums(r)}"),
    ]


def fam_hitung_A():
    VERB2 = [("tambahkan", "dengan"), ("kurangi", "dengan"), ("kalikan", "dengan"), ("jumlahkan", "dan")]
    return [
        ("hitung_A1", lambda r: f"berapa {_num_text(r)} {_op(r)} {_num_text(r)}"),
        ("hitung_A2", lambda r: f"hitung {_num_text(r, False)} {_sym(r)} {_num_text(r, False)}"),
        ("hitung_A3", lambda r: f"{_num_text(r)} {_op(r)} {_num_text(r)} hasilnya berapa{_pick(r, ['', '?'])}"),
        ("hitung_A4", lambda r: (lambda v: f"{v[0]} {_num_text(r)} {v[1]} {_num_text(r)}")(_pick(r, VERB2))),
        ("hitung_A5", lambda r: f"total dari {_nums(r, r.randint(2, 4), ' dan ')} berapa"),
        ("hitung_A6", lambda r: f"kalau {_num_text(r)} {_op(r)} {_num_text(r)} jadi berapa"),
        ("hitung_A7", lambda r: f"{_num_text(r, False)} {_sym(r)} {_num_text(r, False)} = ?"),
        ("hitung_A8", lambda r: f"tolong hitung {_num_text(r)} {_op(r)} {_num_text(r)}"),
    ]


def fam_hitung_B():
    return [
        ("hitung_B1", lambda r: f"sisa {_num_text(r)} setelah dikurangi {_num_text(r)}?"),
        ("hitung_B2", lambda r: f"berapa jumlahnya kalau {_num_text(r)} digabung {_num_text(r)}"),
        ("hitung_B3", lambda r: f"{_num_text(r)} lalu {_op(r)} {_num_text(r)}, berapa jadinya"),
        ("hitung_B4", lambda r: f"kira-kira {_num_text(r, False)} {_sym(r)} {_num_text(r, False)} itu berapa?"),
        ("hitung_B5", lambda r: f"bantu hitungkan {_num_text(r)} {_op(r)} {_num_text(r)}"),
        ("hitung_B6", lambda r: f"hasil dari {_num_text(r)} {_op(r)} {_num_text(r)} apa"),
    ]


def fam_tanggal_A():
    return [
        ("tanggal_A1", lambda r: f"hari apa {_date(r)}"),
        ("tanggal_A2", lambda r: f"{_date(r)} itu hari apa{_pick(r, ['', '?'])}"),
        ("tanggal_A3", lambda r: f"{_date(r)} jatuh pada hari apa"),
        ("tanggal_A4", lambda r: f"selisih {_date(r)} dan {_date(r)}"),
        ("tanggal_A5", lambda r: f"berapa hari dari {_date(r)} sampai {_date(r)}"),
        ("tanggal_A6", lambda r: f"jarak antara {_date(r)} dan {_date(r)} berapa hari"),
        ("tanggal_A7", lambda r: f"berapa lama dari {_date(r)} hingga {_date(r)}"),
    ]


def fam_tanggal_B():
    return [
        ("tanggal_B1", lambda r: f"{_date(r)} hari apakah ya"),
        ("tanggal_B2", lambda r: f"rentang {_date(r)} sampai {_date(r)} ada berapa hari"),
        ("tanggal_B3", lambda r: f"beda {_date(r)} dengan {_date(r)} berapa hari"),
        ("tanggal_B4", lambda r: f"mau tahu {_date(r)} jatuh di hari apa"),
        ("tanggal_B5", lambda r: f"sudah berapa hari sejak {_date(r)} sampai {_date(r)}"),
    ]


def _q_basic(r):
    n, org, ev, it = _pick(r, NAMES), _pick(r, ORGS), _pick(r, EVENTS), _pick(r, ITEMS)
    return _pick(r, [
        f"Di mana {n} tinggal?", f"{n} bekerja di mana?", f"Apa pekerjaan {n}?", f"Kapan {n} lahir?", f"Berapa nomor telepon {n}?", f"Apa alamat {n}?",
        f"Siapa istri {n}?", f"Siapa ketua {org}?", f"Kapan {ev} dimulai?", f"Di ruang mana {ev} berlangsung?", f"Berapa harga {it}?", f"Apa warna {it} {n}?",
        f"Jam berapa {ev} dimulai?", f"Hari apa {ev} diadakan?", f"Berapa jumlah anggota {org}?", f"Siapa bendahara {org}?", f"Di mana {it} disimpan?",
        f"Bagaimana cara menghubungi {n}?", f"Mengapa {ev} ditunda?", f"Berapa umur {n}?", f"Apa makanan kesukaan {n}?", f"Siapa yang menyimpan {it}?"])


def _jebakan_A(r):
    n, org, ev, it, doc = _pick(r, NAMES), _pick(r, ORGS), _pick(r, EVENTS), _pick(r, ITEMS), _pick(r, OBJ_DOC)
    verb = _pick(r, ["menyusun", "membuat", "menulis", "mengubah", "menghitung", "mengurutkan", "merangkum", "menerjemahkan", "mencatat"])
    return _pick(r, [
        f"Siapa yang {verb} {doc}?", f"Siapa yang bertugas {verb} {doc}?", f"Berapa {_pick(r, ['jumlah', 'total', 'selisih'])} {_pick(r, ['peserta', 'biaya', 'anggota', 'tamu'])} {ev}?",
        f"Hari apa {n} {_pick(r, ['libur', 'piket', 'praktik', 'rapat', 'masuk'])}?", f"Kapan batas waktu {_pick(r, ['penyusunan', 'penulisan', 'pembuatan'])} {doc}?",
        f"Di mana {doc} {_pick(r, ['disusun', 'ditulis', 'dibuat', 'disimpan'])}?", f"Apa isi catatan {r.randint(1, 40)}?", f"Catatan mana yang menyebut {_pick(r, NAMES)}?",
        f"Kurang berapa {_pick(r, ['anggota', 'kursi', 'suara'])} untuk {_pick(r, ['kuorum', 'rapat', 'pemilihan'])}?", f"Apa arti kata {_pick(r, WORDS)}?",
        f"Berapa kali {n} {_pick(r, ['menelepon', 'datang', 'izin'])}?", f"Kapan {n} {_pick(r, ['menulis', 'membuat', 'menyusun'])} {doc}?",
        f"Jumlah {_pick(r, ['peserta', 'anggota', 'hadirin'])} {ev} berapa?", f"Total biaya {ev} berapa?", f"Siapa yang menghitung {_pick(r, ['kas', 'iuran', 'hasil lomba'])}?"])


def _jebakan_B(r):
    n, org, ev, doc = _pick(r, NAMES), _pick(r, ORGS), _pick(r, EVENTS), _pick(r, OBJ_DOC)
    return _pick(r, [
        f"Siapa penanggung jawab {_pick(r, ['penyusunan', 'penulisan', 'perhitungan', 'pengurutan'])} {doc}?", f"Tolong cek catatan: siapa yang membuat {doc}?",
        f"{doc} itu disusun {_pick(r, ['oleh siapa', 'kapan', 'di mana'])}?", f"Hari apa saja {org} {_pick(r, ['berlatih', 'buka', 'berkumpul'])}?",
        f"Berapa total {_pick(r, ['hadirin', 'kursi', 'iuran', 'donasi'])} {ev} kemarin?", f"{n} terakhir {_pick(r, ['menulis', 'menghitung', 'menyusun'])} apa?",
        f"Selisih umur {n} dan {_pick(r, NAMES)} berapa tahun menurut catatan?", f"Apa yang tertulis tentang {doc} {org}?",
        f"Sudah berapa lama {n} {_pick(r, ['bekerja', 'tinggal', 'mengajar'])} di {_pick(r, ['sini', 'sekolah itu', 'kantor desa'])}?",
        f"Kapan {org} {_pick(r, ['merangkum', 'menerjemahkan', 'menyusun'])} {doc}?", f"Berapa jumlah {_pick(r, ['catatan', 'dokumen', 'peserta'])} yang sudah masuk?",
        f"Kenapa {doc} {_pick(r, ['diubah', 'ditulis ulang', 'disusun ulang'])}?", f"Bagian mana dari {doc} yang menyebut total {_pick(r, ['biaya', 'anggota', 'peserta'])}?"])


HELLO_A = ["halo", "hai", "hi", "hei", "hello", "selamat pagi", "selamat siang", "selamat sore", "selamat malam", "assalamualaikum", "apa kabar"]
HELLO_TAILS_A = ["", "", "", " kak", " semua", " semuanya", " teman", " pak", " bu", " sob"]
THANKS_A = ["terima kasih", "makasih", "thanks", "terima kasih banyak", "makasih banyak"]
THANKS_TAILS_A = ["", "", " ya", " ya kak", " kak", " atas bantuannya", " sudah membantu", " banyak"]
HELLO_B = ["permisi", "salam", "selamat datang", "wassalamualaikum", "halo selamat pagi", "hai hai", "met pagi", "selamat tinggal", "sampai jumpa", "dadah"]
HELLO_TAILS_B = ["", "", " admin", " bapak ibu", " kawan", " sob", " tim", " dan salam kenal"]
THANKS_B = ["sip makasih", "oke makasih", "thank you", "terima kasih ya", "makasih ya", "thanks a lot"]
THANKS_TAILS_B = ["", "", " atas bantuannya", " kak", " sudah membantu", " admin"]


def _elong(r, w):
    """Panjangkan satu huruf (haiii, halooo, makasihhh); kata yang tidak punya huruf layak dikembalikan apa adanya."""
    idx = [i for i, c in enumerate(w) if c in "aiuoeh" and i > 0]
    if not idx:
        return w
    i = r.choice(idx)
    return w[:i] + w[i] * r.randint(2, 4) + w[i + 1:]


HELLO_NAME = ["halo", "hai", "selamat pagi", "selamat siang", "selamat sore", "selamat malam", "hei"]
EMOJI = [" \U0001F64F", " \U0001F60A", "!", "!!", " :)", " \U0001F44B"]


def fam_sapaan_A():
    return [("sapaan_A1_salam", lambda r: _pick(r, HELLO_A) + _pick(r, HELLO_TAILS_A)),
            ("sapaan_A2_salam_kabar", lambda r: f"{_pick(r, HELLO_A[:10])} {_pick(r, ['apa kabar', 'apa kabarnya'])}"),
            ("sapaan_A3_terima_kasih", lambda r: _pick(r, THANKS_A) + _pick(r, THANKS_TAILS_A)),
            ("sapaan_A4_nama", lambda r: f"{_pick(r, HELLO_NAME)} {_pick(r, NAMES)}"),
            ("sapaan_A5_panjang", lambda r: _elong(r, _pick(r, ["halo", "hai", "hei", "makasih", "thanks"])) + _pick(r, ["", " semua", " kak", " teman", " ya"])),
            ("sapaan_A6_gabungan", lambda r: f"{_pick(r, HELLO_NAME)}, {_pick(r, ['terima kasih', 'makasih'])} {_pick(r, ['ya', 'banyak', 'atas bantuannya', 'sudah membantu'])}"),
            ("sapaan_A7_tanda", lambda r: _pick(r, HELLO_A + THANKS_A) + _pick(r, EMOJI)),
            ("sapaan_A8_islami", lambda r: _pick(r, ["assalamualaikum", "assalamualaikum wr wb", "assalamu alaikum", "assalamualaikum bapak ibu"]) + _pick(r, ["", " semua", " semuanya"]))]


def fam_sapaan_B():
    return [("sapaan_B1_salam", lambda r: _pick(r, HELLO_B) + _pick(r, HELLO_TAILS_B)),
            ("sapaan_B2_salam_ganda", lambda r: f"{_pick(r, ['halo', 'hai', 'selamat pagi'])} {_pick(r, HELLO_TAILS_B[2:])}".strip()),
            ("sapaan_B3_terima_kasih", lambda r: _pick(r, THANKS_B) + _pick(r, THANKS_TAILS_B)),
            ("sapaan_B4_pamit_nama", lambda r: f"{_pick(r, ['sampai jumpa', 'dadah', 'selamat tinggal', 'sampai ketemu lagi', 'permisi dulu'])} {_pick(r, NAMES)}"),
            ("sapaan_B5_gabungan", lambda r: f"{_pick(r, ['permisi', 'salam kenal', 'selamat datang'])}, {_pick(r, ['sip makasih', 'oke makasih', 'terima kasih ya'])}"),
            ("sapaan_B6_panjang", lambda r: _elong(r, _pick(r, ["salam", "permisi", "dadah", "sampai"])) + _pick(r, ["", " semua", " kak", " tim"]))]


CREATE_V_A = ["tuliskan", "tulis", "buatkan", "buat", "karang", "bikin", "susun", "tolong tuliskan", "mohon buatkan"]
CREATE_V_B = ["gubah", "rancang", "kembangkan", "hasilkan", "bantu saya menulis", "bisa tolong buatkan", "saya mau dibuatkan", "coba karangkan", "tolong bantu susun"]
CREATE_OBJ = ["puisi tentang {t}", "cerita pendek tentang {t}", "surat lamaran kerja", "surat izin tidak masuk", "pidato singkat tentang {t}", "email permintaan maaf", "slogan untuk {o}",
              "caption instagram untuk {o}", "lagu tentang {t}", "pantun tentang {t}", "artikel tentang {t}", "dongeng tentang {t}", "daftar belanja mingguan", "kata sambutan untuk {o}",
              "pengumuman untuk {o}", "lelucon lucu"]
SUMMARY_V_A = ["ringkaskan", "rangkum", "tolong ringkaskan", "mohon rangkum"]
SUMMARY_V_B = ["buatkan ringkasan dari", "tolong buat rangkuman", "singkat saja", "padatkan"]
SUMMARY_OBJ = ["rapat kemarin", "isi dokumen tadi", "berita hari ini", "artikel ini", "teks di atas", "cerita tadi", "isi buku itu", "hasil diskusi tadi"]
TRANSLATE_SENT = ["selamat pagi", "terima kasih banyak", "saya mau pergi ke pasar", "apa kabar semuanya", "besok kita bertemu", "tolong bantu saya"]


def _create(r, verbs):
    return f"{_pick(r, verbs)} {_pick(r, CREATE_OBJ).format(t=_pick(r, TOPICS), o=_pick(r, ORGS))}"


def fam_tolak_A():
    return [("tolak_A1", lambda r: _create(r, CREATE_V_A)),
            ("tolak_A2", lambda r: f"bisa {_pick(r, ['tuliskan', 'buatkan', 'karangkan'])} {_pick(r, CREATE_OBJ).format(t=_pick(r, TOPICS), o=_pick(r, ORGS))}?"),
            ("tolak_A3", lambda r: f"terjemahkan ke bahasa {_pick(r, ['Inggris', 'Jawa', 'Sunda', 'Arab'])}: {_pick(r, TRANSLATE_SENT)}"),
            ("tolak_A4", lambda r: f"{_pick(r, SUMMARY_V_A)} {_pick(r, SUMMARY_OBJ)}")]


def fam_tolak_B():
    return [("tolak_B1", lambda r: _create(r, CREATE_V_B)),
            ("tolak_B2", lambda r: f"tolong {_pick(r, ['karangkan', 'tuliskan', 'buatkan'])} {_pick(r, CREATE_OBJ).format(t=_pick(r, TOPICS), o=_pick(r, ORGS))} dong"),
            ("tolak_B3", lambda r: f"{_pick(r, SUMMARY_V_B)} {_pick(r, SUMMARY_OBJ)}"),
            ("tolak_B4", lambda r: f"bagaimana kalau kamu menerjemahkan '{_pick(r, TRANSLATE_SENT)}' ke bahasa {_pick(r, ['Inggris', 'Jawa', 'Prancis'])}")]


def fam_cari_A():
    def greet_q(r):
        return f"{_pick(r, ['halo', 'hai', 'selamat pagi', 'permisi', 'selamat siang'])}, {(lambda q: q[0].lower() + q[1:])(_q_basic(r))}"

    def def_q(r):
        w = _pick(r, WORDS)
        return _pick(r, [f"apa itu {w}", f"{w} adalah apa", f"pengertian {w}", f"apa arti kata {w}", f"jelaskan {w}", f"definisi {w}", f"{w} itu apa?"])

    def gen_q(r):
        return G.make_fact(r, r.choice(G.TRAIN_ATTRS))["question"]

    return [("cari_A1_pembangkit_fakta", gen_q), ("cari_A2_dasar", _q_basic), ("cari_A3_salam_tanya", greet_q), ("cari_A4_definisi", def_q), ("cari_A5_jebakan", _jebakan_A)]


def fam_cari_B():
    def paraphrase(r):
        q = _q_basic(r)
        return _pick(r, [f"tolong cek catatan: {q[0].lower() + q[1:]}", f"{q[:-1]}, ya?", f"kasih tahu dong {q[0].lower() + q[1:-1]}", f"mau tanya, {q[0].lower() + q[1:]}", f"cek: {q[0].lower() + q[1:-1]}"])

    def short_q(r):
        n, ev = _pick(r, NAMES), _pick(r, EVENTS)
        return _pick(r, [f"{n} di mana?", f"alamat {n}?", f"{ev} kapan?", f"telepon {n}?", f"{ev} jam berapa", f"siapa ketua {_pick(r, ORGS)}", f"{n} kerja apa"])

    return [("cari_B1_parafrase", paraphrase), ("cari_B2_pendek", short_q), ("cari_B3_jebakan", _jebakan_B)]


FAMILIES_A = {"urut": fam_urut_A, "hitung": fam_hitung_A, "tanggal": fam_tanggal_A, "cari": fam_cari_A, "sapaan": fam_sapaan_A, "tolak": fam_tolak_A}
FAMILIES_B = {"urut": fam_urut_B, "hitung": fam_hitung_B, "tanggal": fam_tanggal_B, "cari": fam_cari_B, "sapaan": fam_sapaan_B, "tolak": fam_tolak_B}


# ---------- gaya dan gangguan ----------
PREFIX = ["tolong ", "mohon ", "coba ", "bisa ", "kak ", "min ", "permisi, ", "halo, ", "bantu ", "eh "]
SUFFIX = [" ya", " dong", " nih", " deh", " please", " makasih", " ya kak", " ya min"]
COLLOQ = [("tidak", "gak"), ("bagaimana", "gimana"), ("berapa", "brp"), ("sudah", "udah"), ("kenapa", "knp"), ("yang", "yg"), ("dengan", "sama"), ("tolong", "tlg")]


def typo(rng, text):
    words = text.split(" ")
    cand = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    for i in rng.sample(cand, min(len(cand), rng.randint(1, 2))):
        w = words[i]
        k = rng.randint(1, len(w) - 2)
        kind = rng.random()
        words[i] = (w[:k] + w[k + 1] + w[k] + w[k + 2:]) if kind < 0.4 else (w[:k] + w[k + 1:]) if kind < 0.75 else (w[:k] + w[k] + w[k:])
    return " ".join(words)


POLITE = ("tolong", "mohon", "coba", "bisa", "bisakah", "kak", "min", "permisi", "halo", "bantu", "eh", "saya", "bagaimana")


def style(rng, text, strength=1.0, force=False, typo_ok=True, prefix_ok=True, suffix_ok=True, flags=None):
    """Terapkan gaya santai dan gangguan acak. force=True: pastikan sedikitnya satu perubahan (dipakai set berisik)."""
    t, changed = text, False
    if prefix_ok and rng.random() < 0.25 * strength and not t.lower().startswith(POLITE):   # tidak ada awalan ganda
        t, changed = rng.choice(PREFIX) + t, True
    if suffix_ok and rng.random() < 0.15 * strength:
        t, changed = t.rstrip("?.") + rng.choice(SUFFIX), True
    if rng.random() < 0.05 * strength:
        for a, b in COLLOQ:
            if a in t:
                t, changed = t.replace(a, b, 1), True
                break
    if typo_ok and rng.random() < (0.12 if not force else 0.55) * strength:
        t, changed = typo(rng, t), True
        if flags is not None:
            flags["typo"] = True
    if rng.random() < (0.30 if not force else 0.5) * strength:
        t, changed = re.sub(r"[?.!]", "", t), True
    if rng.random() < (0.35 if not force else 0.5) * strength:
        t, changed = t.lower(), True
    out = re.sub(r"\s+", " ", t).strip()
    if force and out == re.sub(r"\s+", " ", text).strip():      # 'changed' bisa menyala tanpa efek (mis. huruf kecil pada teks yang sudah kecil)
        out = typo(rng, text) if rng.random() < 0.5 else rng.choice(PREFIX) + text
        if out == text:
            out = rng.choice(PREFIX) + text
    return out


# ---------- pembangkitan dan kontrol ----------
def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def _shingles(s, n=3):
    s = " " + _norm(s) + " "
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def generate(families, label, n, rng, tag_family=True):
    """n pesan unik untuk satu kelas dari keluarga templat yang diberikan (bobot rata antar templat)."""
    fams = families[label]()
    seen, out, tries, typos = set(), [], 0, 0
    while len(out) < n and tries < n * 60:
        tries += 1
        name, fn = fams[tries % len(fams)] if tries % 3 else rng.choice(fams)
        flags = {}
        # kuota salah ketik 25% per kelas: ruang templat kecil (mis. sapaan) bisa habis versi bersihnya dan sisanya hanya yang bersalah ketik
        msg = style(rng, fn(rng), typo_ok=typos < 0.25 * max(len(out), 20), prefix_ok=(label != "sapaan"), suffix_ok=(label != "sapaan"), flags=flags)
        key = _norm(msg)
        if key in seen or not key:
            continue
        seen.add(key)
        typos += bool(flags.get("typo"))
        out.append({"message": msg, "label": label, "family": name, "salah_ketik": bool(flags.get("typo"))})
    return out


def length_bucket(msg):
    k = len(msg.split())
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= k <= hi:
            return i
    return len(BUCKETS) - 1


def balance_lengths(pools, target_per_class, rng):
    """Cuplik tiap kelas agar sebaran panjang kata mendekati BUCKET_WEIGHT (tidak melebihi yang tersedia)."""
    out = {}
    for label, items in pools.items():
        by_b = collections.defaultdict(list)
        for it in items:
            by_b[length_bucket(it["message"])].append(it)
        pick = []
        for b, w in enumerate(BUCKET_WEIGHT):
            avail = by_b.get(b, [])
            rng.shuffle(avail)
            pick += avail[:int(round(w * target_per_class[label]))]
        out[label] = pick
    return out


def drop_near_duplicates(train, protected, threshold=0.8):
    """Buang pesan latih yang identik atau hampir sama (Jaccard 3-gram >= threshold) dengan pesan di set uji (protected)."""
    exact = {_norm(m) for m in protected}
    sh = [_shingles(m) for m in protected]
    index = collections.defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            index[g].append(i)
    kept, dropped = [], 0
    for it in train:
        key = _norm(it["message"])
        if key in exact:
            dropped += 1
            continue
        s = _shingles(it["message"])
        hits = collections.Counter(i for g in s for i in index.get(g, ()))
        bad = any(c / (len(s) + len(sh[i]) - c) >= threshold for i, c in hits.items() if c >= 3)
        if bad:
            dropped += 1
        else:
            kept.append(it)
    return kept, dropped


def build(seed=42, n_train=3000, n_train_cari=10000, n_valid=300, n_valid_cari=600, n_jebakan=150, frozen=None):
    """frozen: dict berisi 'validasi', 'jebakan', 'berisik' yang sudah dibekukan; bila ada, dipakai apa adanya (tidak dibangkitkan ulang)."""
    frozen = frozen or {}
    rng = random.Random(seed)
    acceptance = {}
    for f in ("memory_router.jsonl", "memory_router_heldout.jsonl", "memory_router_test2.jsonl"):
        acceptance[f] = [json.loads(l) for l in (PROJECT_ROOT / "evalsets" / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    acc_msgs = [r["message"] for rows in acceptance.values() for r in rows]
    # validasi: keluarga B (tak ada di latih), seluruh kelas; jebakan uji: keluarga B jebakan saja
    valid = frozen.get("validasi") or [it for lab in LABELS for it in generate(FAMILIES_B, lab, n_valid_cari if lab == "cari" else n_valid, rng)]
    jeb_fn = _jebakan_B
    seen, jebakan = set(), list(frozen.get("jebakan") or [])
    while len(jebakan) < n_jebakan:
        msg = style(rng, jeb_fn(rng), strength=0.7)
        if _norm(msg) not in seen:
            seen.add(_norm(msg))
            jebakan.append({"message": msg, "label": "cari", "family": "cari_B3_jebakan"})
    # latih: keluarga A, kumpulkan lebih banyak lalu seimbangkan panjang
    target = {lab: (n_train_cari if lab == "cari" else n_train) for lab in LABELS}
    pools = {lab: generate(FAMILIES_A, lab, int(target[lab] * 2.2), rng) for lab in LABELS}
    protected = acc_msgs + [it["message"] for it in valid + jebakan]
    dropped_total = 0
    for lab in LABELS:
        pools[lab], d = drop_near_duplicates(pools[lab], protected)
        dropped_total += d
    train_by = balance_lengths(pools, target, rng)
    train = [it for lab in LABELS for it in train_by[lab]]
    rng.shuffle(train)
    # berisik: varian bergangguan dari set uji resmi (label sama)
    noisy = list(frozen.get("berisik") or [])
    for rows in ([] if noisy else acceptance.values()):
        for r in rows:
            for _ in range(3):
                noisy.append({"message": style(rng, r["message"], strength=1.0, force=True), "label": r["label"], "family": "berisik", "asal": r["message"]})
    return {"train": train, "validasi": valid, "jebakan": jebakan, "berisik": noisy, "dibuang_hampir_sama": dropped_total}


def length_stats(rows):
    out = {}
    for lab in LABELS:
        L = sorted(len(r["message"].split()) for r in rows if r["label"] == lab)
        if L:
            out[lab] = {"n": len(L), "median": L[len(L) // 2], "rata": round(sum(L) / len(L), 2), "min": L[0], "maks": L[-1]}
    return out


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


REVIEW_HEAD = """# Sampel label untuk diperiksa (tahap 1, pemilih alat)
Periksa apakah **label** tiap pesan benar menurut Anda. Isi kolom "Benar?" dengan **ya**, **tidak**, atau **ragu** (ragu = pesan bisa dua kelas).
Kelas: `urut` (mengurutkan angka), `hitung` (operasi hitung), `tanggal` (hari atau selisih tanggal), `cari` (pertanyaan tentang catatan, termasuk definisi),
`sapaan` (menyapa, terima kasih), `tolak` (menulis, menerjemahkan, merangkum: permintaan yang tidak didukung). Target: ketidaksepakatan <= 2%.

| No | Pesan | Label | Benar? |
|---|---|---|---|
"""


def write_review(path, build_out, rng, per_class=25):
    pool = build_out["train"] + build_out["validasi"] + build_out["jebakan"]
    rows = []
    for lab in LABELS:
        items = [r for r in pool if r["label"] == lab]
        rows += rng.sample(items, min(per_class, len(items)))
    rng.shuffle(rows)
    lines = [f"| {i} | {r['message'].replace('|', '/')} | {r['label']} | |" for i, r in enumerate(rows[:150], 1)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(REVIEW_HEAD + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "router"))
    ap.add_argument("--evalsets-dir", default=str(PROJECT_ROOT / "evalsets"))
    ap.add_argument("--refresh-evalsets", action="store_true", help="bangkitkan ulang dan TIMPA set uji yang sudah dibekukan (jarang diperlukan)")
    args = ap.parse_args()
    out, ev = Path(args.out_dir), Path(args.evalsets_dir)
    names = {"validasi": "router_validasi.jsonl", "jebakan": "router_jebakan.jsonl", "berisik": "router_berisik.jsonl"}
    frozen = {k: [json.loads(l) for l in (ev / f).read_text(encoding="utf-8").splitlines() if l.strip()] for k, f in names.items()
              if (ev / f).exists() and not args.refresh_evalsets}
    b = build(args.seed, frozen=frozen)
    if frozen:
        print("set uji yang sudah dibekukan dimuat apa adanya:", sorted(frozen))
    write_jsonl(out / "train.jsonl", b["train"])
    for k, f in names.items():
        if k not in frozen:
            write_jsonl(ev / f, b[k])
    summary = {"seed": args.seed, "jumlah": {k: len(v) for k, v in b.items() if isinstance(v, list)}, "dibuang_hampir_sama": b["dibuang_hampir_sama"],
               "per_kelas_train": dict(collections.Counter(r["label"] for r in b["train"])), "per_kelas_validasi": dict(collections.Counter(r["label"] for r in b["validasi"])),
               "panjang_train": length_stats(b["train"]), "panjang_validasi": length_stats(b["validasi"]),
               "keluarga_train": sorted({r["family"] for r in b["train"]}), "keluarga_validasi": sorted({r["family"] for r in b["validasi"]})}
    (out / "ringkasan.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    write_review(out / "ulasan_label_150.md", b, random.Random(args.seed + 1))
    print(json.dumps({k: summary[k] for k in ("jumlah", "dibuang_hampir_sama", "per_kelas_train")}, ensure_ascii=False))
    print("panjang kata (median) train :", {k: v["median"] for k, v in summary["panjang_train"].items()})
    print("panjang kata (median) valid.:", {k: v["median"] for k, v in summary["panjang_validasi"].items()})


if __name__ == "__main__":
    main()
