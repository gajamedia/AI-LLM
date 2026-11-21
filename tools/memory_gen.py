"""
Generator fakta/catatan sintetis untuk SFT "memori" dan pengujiannya.

Setiap atribut punya pola kalimat catatan dan pertanyaan:
  A = dipakai untuk melatih;  B = DITAHAN, hanya untuk uji (kalimat yang tidak pernah dilatih).
Atribut 'hobi' dan 'umur' (TEST_ONLY) tidak pernah muncul di data latih.
Nilai memakai nama fiktif agar jawaban tidak mungkin berasal dari pengetahuan hafalan model.
"""
import random

FIRST = ("Sari Budi Dewi Agus Rina Joko Wati Hendra Lestari Bayu Putri Rahmat Maya Dedi Ani Eko Tuti Fajar Indah Ketut "
         "Nur Yusuf Citra Andi Mega Surya Lina Rudi Ayu Taufik Wahyu Yanti Galih Dian Tri Bagus Intan Reza").split()
SYL = ["su", "ka", "ra", "ma", "ti", "ba", "lu", "ge", "no", "pa", "di", "ja", "wi", "sa", "ke", "mu", "la", "ro", "ne", "ho"]
JOBS = ["dokter gigi", "tukang kayu", "guru olahraga", "pilot", "penjahit", "petani kopi", "juru masak", "arsitek",
        "nelayan", "montir", "apoteker", "penerjemah", "fotografer", "pengacara", "perawat"]
FOODS = ["gudeg", "rawon", "pempek", "rendang", "sate padang", "soto betawi", "bakso", "gado-gado", "ayam betutu",
         "tahu campur", "nasi liwet", "mie aceh", "coto makassar", "pecel", "lontong sayur"]
COLORS = ["merah", "biru", "hijau", "kuning", "hitam", "putih", "oranye", "ungu", "abu-abu", "cokelat"]
ANIMALS = ["kucing", "anjing", "burung beo", "kelinci", "kura-kura", "hamster"]
MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
HOBI = ["memancing", "melukis", "bersepeda", "membaca", "berkebun", "fotografi", "mendaki gunung", "bermain catur"]


def word(rng, n):
    return "".join(rng.choice(SYL) for _ in range(n))


def person_name(rng, first=None):
    return f"{first or rng.choice(FIRST)} {word(rng, 2).capitalize()}{word(rng, 1)}"


def _city(rng): return word(rng, 3).capitalize()
def _year(rng): return str(rng.randint(1950, 2005))
def _job(rng): return rng.choice(JOBS)
def _food(rng): return rng.choice(FOODS)
def _pet(rng): return word(rng, 2).capitalize()
def _floor(rng): return str(rng.randint(2, 30))
def _color(rng): return rng.choice(COLORS)
def _date(rng): return f"{rng.randint(1, 28)} {rng.choice(MONTHS)}"
def _price(rng): return str(rng.randint(5, 95))
def _phone(rng): return "08" + "".join(str(rng.randint(0, 9)) for _ in range(10))
def _street(rng): return f"Jalan {word(rng, 2).capitalize()} nomor {rng.randint(1, 120)}"
def _hobi(rng): return rng.choice(HOBI)
def _age(rng): return str(rng.randint(19, 75))


def _proj(rng): return f"Proyek {word(rng, 2).capitalize()}-{rng.randint(2, 99)}"
def _item(rng): return word(rng, 3).capitalize()
def _team(rng): return word(rng, 2).capitalize() + word(rng, 1)


# subjek: 'person' | 'project' | 'item' | 'team'; nilai: fungsi pembuat; A/B = pola catatan (n) dan pertanyaan (q)
ATTRS = {
    "lahir_kota": dict(subj="person", val=_city,
        nA=["{S} lahir di kota {V}.", "Kota kelahiran {S} adalah {V}.", "{S} dilahirkan di {V}.", "Tempat lahir {S} ialah {V}."],
        nB=["{S} lahir dan dibesarkan di {V}.", "Di {V} lah {S} dilahirkan."],
        qA=["Di kota mana {S} dilahirkan?", "Di mana {S} lahir?", "Apa kota kelahiran {S}?", "{S} lahir di kota apa?"],
        qB=["Kota apa yang menjadi tempat kelahiran {S}?", "Sebutkan tempat {S} dilahirkan."]),
    "lahir_tahun": dict(subj="person", val=_year,
        nA=["{S} lahir pada tahun {V}.", "Tahun kelahiran {S} adalah {V}.", "{S} dilahirkan tahun {V}.", "Kelahiran {S} terjadi di tahun {V}."],
        nB=["{S} lahir di tahun {V}, menurut catatan keluarga.", "Sejak {V}, {S} sudah ada di dunia ini."],
        qA=["Pada tahun berapa {S} dilahirkan?", "Kapan tahun lahir {S}?", "{S} lahir tahun berapa?", "Tahun kelahiran {S} kapan?"],
        qB=["Di tahun berapa {S} lahir?", "Sebutkan tahun {S} dilahirkan."]),
    "profesi": dict(subj="person", val=_job,
        nA=["{S} bekerja sebagai {V}.", "Pekerjaan {S} adalah {V}.", "{S} berprofesi sebagai {V}.", "Profesi {S} ialah {V}."],
        nB=["Sehari-hari {S} menjalani pekerjaan sebagai {V}.", "{S} menekuni dunia sebagai {V}."],
        qA=["Apa pekerjaan {S}?", "{S} bekerja sebagai apa?", "Apa profesi {S}?", "Pekerjaan {S} apa?"],
        qB=["Sebagai apa {S} bekerja sehari-hari?", "Apa yang ditekuni {S} sebagai profesi?"]),
    "makanan": dict(subj="person", val=_food,
        nA=["Makanan favorit {S} adalah {V}.", "{S} paling suka makan {V}.", "Makanan kesukaan {S} ialah {V}.", "{S} gemar menyantap {V}."],
        nB=["Kalau soal makanan, {S} selalu memilih {V}.", "Hidangan yang paling disukai {S} yaitu {V}."],
        qA=["Apa makanan kesukaan {S}?", "{S} paling suka makan apa?", "Apa makanan favorit {S}?", "Makanan apa yang digemari {S}?"],
        qB=["Hidangan apa yang paling disukai {S}?", "Apa pilihan makanan {S}?"]),
    "hewan": dict(subj="person", val=_pet,
        nA=["{S} memelihara seekor hewan bernama {V}.", "Hewan peliharaan {S} bernama {V}.", "{S} punya hewan kesayangan bernama {V}.", "Nama hewan peliharaan {S} adalah {V}."],
        nB=["Di rumah, {S} merawat hewan yang diberi nama {V}.", "Peliharaan milik {S} dipanggil {V}."],
        qA=["Siapa nama hewan peliharaan {S}?", "Apa nama hewan kesayangan {S}?", "Hewan peliharaan {S} bernama siapa?", "Nama peliharaan {S} apa?"],
        qB=["Dengan nama apa peliharaan {S} dipanggil?", "Sebutkan nama hewan yang dirawat {S}."]),
    "lantai": dict(subj="person", val=_floor,
        nA=["Ruang kerja {S} berada di lantai {V}.", "{S} berkantor di lantai {V}.", "Kantor {S} terletak di lantai {V}.", "Lantai ruang kerja {S} adalah {V}."],
        nB=["Untuk menemui {S}, naiklah ke lantai {V}.", "Meja kerja {S} ada di lantai {V} gedung itu."],
        qA=["Di lantai berapa ruang kerja {S}?", "{S} berkantor di lantai berapa?", "Kantor {S} ada di lantai berapa?", "Lantai berapa ruang kerja {S}?"],
        qB=["Ke lantai berapa kita harus naik untuk bertemu {S}?", "Meja kerja {S} terletak di lantai berapa?"]),
    "mobil": dict(subj="person", val=_color,
        nA=["{S} mengendarai mobil berwarna {V}.", "Mobil milik {S} berwarna {V}.", "Warna mobil {S} adalah {V}.", "{S} punya mobil {V}."],
        nB=["Kendaraan roda empat {S} dicat warna {V}.", "Orang-orang mengenal {S} lewat mobilnya yang berwarna {V}."],
        qA=["Apa warna mobil {S}?", "Mobil {S} berwarna apa?", "{S} mengendarai mobil warna apa?", "Warna kendaraan {S} apa?"],
        qB=["Dicat dengan warna apa mobil {S}?", "Mobil {S} tampak berwarna apa?"]),
    "tenggat": dict(subj="project", val=_date,
        nA=["Tenggat {S} jatuh pada tanggal {V}.", "{S} harus selesai pada {V}.", "Batas waktu {S} adalah {V}.", "Deadline {S} yaitu tanggal {V}."],
        nB=["Pekerjaan {S} perlu rampung paling lambat {V}.", "Kapan pun terjadi, {S} wajib beres pada {V}."],
        qA=["Kapan batas waktu {S}?", "Kapan tenggat {S}?", "{S} harus selesai kapan?", "Tanggal berapa deadline {S}?"],
        qB=["Paling lambat kapan {S} harus rampung?", "Kapan {S} wajib beres?"]),
    "harga": dict(subj="item", val=_price,
        nA=["Harga satu kotak {S} adalah {V} ribu rupiah.", "Satu kotak {S} dijual seharga {V} ribu rupiah.", "{S} per kotak harganya {V} ribu rupiah.", "Untuk satu kotak {S}, harganya {V} ribu rupiah."],
        nB=["Pembeli cukup membayar {V} ribu rupiah untuk sekotak {S}.", "Sekotak {S} dibanderol {V} ribu rupiah."],
        qA=["Berapa harga satu kotak {S}?", "Satu kotak {S} harganya berapa?", "{S} per kotak berapa harganya?", "Berapa ribu rupiah satu kotak {S}?"],
        qB=["Berapa yang harus dibayar untuk sekotak {S}?", "Berapa banderol sekotak {S}?"]),
    "ketua": dict(subj="team", val=person_name, val_is_name=True,
        nA=["{V} adalah ketua tim {S}.", "Ketua tim {S} ialah {V}.", "Tim {S} dipimpin oleh {V}.", "{V} memimpin tim {S}."],
        nB=["Pemimpin tim {S} dipegang oleh {V}.", "Kalau bicara tim {S}, orang yang menjadi ketuanya adalah {V}."],
        qA=["Siapa ketua tim {S}?", "Tim {S} dipimpin oleh siapa?", "Siapa yang memimpin tim {S}?", "Ketua tim {S} siapa?"],
        qB=["Siapa pemimpin tim {S}?", "Siapa yang menjadi ketua dalam tim {S}?"]),
    "telepon": dict(subj="person", val=_phone,
        nA=["Nomor telepon {S} adalah {V}.", "{S} bisa dihubungi di nomor {V}.", "Kontak {S} yaitu {V}.", "Nomor ponsel {S} ialah {V}."],
        nB=["Untuk menghubungi {S}, telepon ke {V}.", "Nomor yang dipakai {S} adalah {V}."],
        qA=["Berapa nomor telepon {S}?", "{S} bisa dihubungi di nomor berapa?", "Apa nomor ponsel {S}?", "Nomor kontak {S} berapa?"],
        qB=["Ke nomor berapa kita harus menelepon {S}?", "Nomor apa yang dipakai {S}?"]),
    "alamat": dict(subj="person", val=_street,
        nA=["{S} tinggal di {V}.", "Alamat rumah {S} adalah {V}.", "Rumah {S} berada di {V}.", "{S} bertempat tinggal di {V}."],
        nB=["Rumah {S} dapat ditemukan di {V}.", "Kediaman {S} ada di {V}."],
        qA=["Di mana {S} tinggal?", "Apa alamat rumah {S}?", "Di mana rumah {S}?", "{S} tinggal di jalan apa?"],
        qB=["Di jalan mana kediaman {S}?", "Ke alamat mana kita menemui {S}?"]),
    # --- hanya untuk uji (atribut tak pernah dilatih) ---
    "hobi": dict(subj="person", val=_hobi, test_only=True,
        nA=["{S} gemar {V}.", "Hobi {S} adalah {V}.", "{S} menghabiskan waktu luang dengan {V}."],
        nB=["Di akhir pekan, {S} biasanya {V}."],
        qA=["Apa hobi {S}?", "{S} gemar melakukan apa?"],
        qB=["Apa yang dilakukan {S} di waktu luang?"]),
    "umur": dict(subj="person", val=_age, test_only=True,
        nA=["{S} berusia {V} tahun.", "Umur {S} adalah {V} tahun.", "{S} sekarang genap {V} tahun."],
        nB=["Usia {S} saat ini {V} tahun."],
        qA=["Berapa umur {S}?", "{S} berusia berapa tahun?"],
        qB=["Berapa usia {S} sekarang?"]),
}
TRAIN_ATTRS = [k for k, v in ATTRS.items() if not v.get("test_only")]
SUBJ_FN = {"person": person_name, "project": _proj, "item": _item, "team": _team}


def new_subject(rng, kind, first=None):
    return person_name(rng, first) if kind == "person" else SUBJ_FN[kind](rng)


def make_fact(rng, attr, subject=None, split="A", first=None):
    """Kembalikan dict(note, question, answer, attr, subject, value)."""
    a = ATTRS[attr]
    s = subject or new_subject(rng, a["subj"], first)
    v = a["val"](rng)
    n = rng.choice(a["nA"] if split == "A" else a["nB"])
    q = rng.choice(a["qA"] if split == "A" else a["qB"])
    return {"note": n.format(S=s, V=v), "question": q.format(S=s), "answer": v, "attr": attr, "subject": s, "value": v}


def _distractors(rng, fact, k, attrs, split):
    a = ATTRS[fact["attr"]]
    out = []
    kinds = ["same_subject", "same_attr", "random"]
    for _ in range(k):
        kind = rng.choice(kinds)
        if kind == "same_subject":
            others = [x for x in attrs if x != fact["attr"] and ATTRS[x]["subj"] == a["subj"]]
            if others:
                out.append(make_fact(rng, rng.choice(others), subject=fact["subject"], split=split)["note"]); continue
        if kind == "same_attr" and a["subj"] == "person":
            out.append(make_fact(rng, fact["attr"], first=fact["subject"].split()[0], split=split)["note"]); continue
        out.append(make_fact(rng, rng.choice(attrs), split=split)["note"])
    return out


def make_example(rng, attrs=None, split="A"):
    """Satu contoh: dict(ctx, question, answer, kind). kind: answerable|unanswerable_person|unanswerable_attr|correction"""
    attrs = attrs or TRAIN_ATTRS
    r = rng.random()
    attr = rng.choice(attrs)
    fact = make_fact(rng, attr, split=split)
    if r < 0.07:  # koreksi: catatan lama + "Pembaruan:" dengan nilai baru
        new = make_fact(rng, attr, subject=fact["subject"], split=split)
        while new["value"] == fact["value"]:
            new = make_fact(rng, attr, subject=fact["subject"], split=split)
        notes = [fact["note"], "Pembaruan: " + new["note"]]
        extra = _distractors(rng, fact, rng.randint(0, 1), attrs, split)
        notes = notes + extra
        return {"ctx": " ".join(notes), "question": fact["question"], "answer": new["value"], "kind": "correction"}
    if r < 0.27:  # tak terjawab
        k = rng.randint(1, 3)
        if rng.random() < 0.5 and ATTRS[attr]["subj"] == "person":
            ghost = make_fact(rng, attr, first=fact["subject"].split()[0], split=split)  # orang lain, nama depan sama
            notes = [fact["note"]] + _distractors(rng, fact, k - 1, attrs, split)
            rng.shuffle(notes)
            return {"ctx": " ".join(notes), "question": ghost["question"], "answer": None, "kind": "unanswerable_person"}
        others = [x for x in attrs if x != attr and ATTRS[x]["subj"] == ATTRS[attr]["subj"]]
        if others:
            other = make_fact(rng, rng.choice(others), subject=fact["subject"], split=split)
            notes = [other["note"]] + [d for d in _distractors(rng, other, k - 1, attrs, split)]
            rng.shuffle(notes)
            return {"ctx": " ".join(notes), "question": fact["question"], "answer": None, "kind": "unanswerable_attr"}
    k = rng.choice([0, 0, 1, 2, 2])  # jumlah pengecoh
    notes = [fact["note"]] + _distractors(rng, fact, k, attrs, split)
    rng.shuffle(notes)
    return {"ctx": " ".join(notes), "question": fact["question"], "answer": fact["answer"], "kind": "answerable"}
