"""
Generator kompositional v2: kalimat catatan rumit (banyak fakta, sebab-akibat, peran/penggantian,
relasi keluarga, daftar, jumlah dan harga, jadwal, penerimaan). Tiap 'keluarga' menghasilkan satu catatan
dengan beberapa pasangan (pertanyaan, jawaban) + pertanyaan tak terjawab yang masih berkaitan.
Dirancang terpisah dari set uji tulisan tangan (data/sft/memory_complex*.jsonl): jangan menyalin kalimat uji.
"""
import random

import memory_gen as G

DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
EVENTS = ["rapat", "pelatihan", "ujian", "bazar", "kunjungan", "arisan", "lomba", "pertemuan"]
TOPICS = ["produk", "keuangan", "kebersihan", "penjualan", "keamanan", "pendidikan", "kesehatan", "olahraga", "seni", "logistik"]
ROOMS = ["Melati", "Anggrek", "Mawar", "Teratai", "Dahlia", "Kenanga", "Flamboyan"]
CAUSES = ["hujan deras", "pemadaman listrik", "banjir", "kerusakan mesin", "sakit demam", "kemacetan panjang", "pohon tumbang", "gangguan sinyal"]
ITEMS = ["kursi kayu", "meja lipat", "kardus buku", "karung beras", "lampu hias", "kipas angin", "rak besi", "ember plastik", "payung besar"]
UNITS = ["buah", "karung", "dus", "lusin"]
ROLES = ["bendahara", "ketua panitia", "kepala bagian", "sekretaris", "koordinator", "manajer gudang"]
ALLERGY = ["udang", "kacang tanah", "telur", "susu sapi", "ikan laut", "gandum", "kerang"]
CONTAINERS = ["kotak kayu", "lemari besi", "laci meja", "tas kain", "map biru", "peti tua", "rak atas"]
PLACES = ["balai desa", "kantor pos", "gedung serbaguna", "aula sekolah", "pasar pagi", "terminal", "stasiun", "taman kota"]
SERVICES = ["bus", "kapal feri", "kereta", "angkot", "kapal cepat"]
SHOP = ["Sumber", "Berkah", "Maju", "Sentosa", "Makmur", "Harapan", "Sejahtera"]
FOODS2 = ["nasi goreng", "mie ayam", "soto", "pecel lele", "bubur", "sate ayam", "ketoprak"]
SMALL = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]


def num_words(n):
    """Angka 1..999 -> kata Indonesia."""
    if n < 10:
        return SMALL[n]
    if n == 10:
        return "sepuluh"
    if n == 11:
        return "sebelas"
    if n < 20:
        return SMALL[n - 10] + " belas"
    if n < 100:
        t, u = divmod(n, 10)
        return SMALL[t] + " puluh" + (" " + SMALL[u] if u else "")
    h, r = divmod(n, 100)
    head = "seratus" if h == 1 else SMALL[h] + " ratus"
    return head + (" " + num_words(r) if r else "")


def price_words(rng):
    k = rng.choice([5, 8, 12, 15, 20, 25, 35, 45, 60, 75, 85, 120, 150, 250])
    return f"{num_words(k)} ribu", k


def count_words(rng, lo=2, hi=90):
    n = rng.randint(lo, hi)
    return num_words(n), n


def pick(rng, xs):
    return rng.choice(xs)


def _person(rng):
    return f"{rng.choice(['Bu', 'Pak', 'Kak', 'Mas'])} {rng.choice(G.FIRST)}" if rng.random() < 0.5 else G.person_name(rng)


def _topic_event(rng):
    return f"{pick(rng, EVENTS)} {pick(rng, TOPICS)}"


def fam_schedule(rng):
    ev, day, room = _topic_event(rng), pick(rng, DAYS), pick(rng, ROOMS)
    hour = rng.randint(7, 17)
    t = f"{hour}.{rng.choice(['00', '30'])}"
    note = rng.choice([
        f"{ev.capitalize()} dijadwalkan ulang ke hari {day} pukul {t} di ruang {room}.",
        f"Karena ada perubahan, {ev} kini berlangsung pada hari {day}, mulai pukul {t}, bertempat di ruang {room}.",
        f"Panitia memutuskan {ev} dipindah ke ruang {room} pada hari {day} dengan waktu mulai pukul {t}."])
    qa = [(rng.choice([f"Kapan {ev} diadakan?", f"Pada hari apa {ev} berlangsung?"]), day),
          (rng.choice([f"Jam berapa {ev} dimulai?", f"Pukul berapa {ev} mulai?"]), t),
          (rng.choice([f"Di ruang mana {ev} diadakan?", f"Di mana {ev} berlangsung?"]), room)]
    un = [f"Siapa yang memimpin {ev}?", f"Berapa orang yang hadir pada {ev}?"]
    return note, qa, un


def fam_two_facts(rng):
    s = G.person_name(rng)
    job, city = pick(rng, G.JOBS), G.word(rng, 3).capitalize()
    year, street = str(rng.randint(1960, 2003)), G._street(rng)
    note, qa, un = rng.choice([
        (f"{s} bekerja sebagai {job} di kota {city}, dan sehari-hari ia pulang ke rumahnya di {street}.",
         [(f"Apa pekerjaan {s}?", job), (f"Di kota mana {s} bekerja?", city), (f"Di mana rumah {s}?", street)],
         [f"Berapa gaji {s}?", f"Siapa atasan {s}?"]),
        (f"{s} lahir di kota {city} pada tahun {year}, kemudian merantau dan menjadi {job}.",
         [(f"Di kota mana {s} lahir?", city), (f"Pada tahun berapa {s} lahir?", year), (f"Apa pekerjaan {s} sekarang?", job)],
         [f"Siapa nama istri {s}?", f"Apa hobi {s}?"]),
        (f"Selain menjadi {job}, {s} yang berasal dari {city} dikenal gemar menyantap {pick(rng, FOODS2)}.",
         [(f"Apa pekerjaan {s}?", job), (f"Dari mana asal {s}?", city)],
         [f"Berapa umur {s}?"])])
    return note, qa, un


def fam_order(rng):
    p = _person(rng)
    item, unit = pick(rng, ITEMS), pick(rng, UNITS)
    cw, cn = count_words(rng)
    pw, _ = price_words(rng)
    city = G.word(rng, 3).capitalize()
    note = rng.choice([
        f"{p} memesan {cw} {unit} {item} dari pengrajin di {city} dengan harga {pw} rupiah per {unit}.",
        f"Dari {city}, {p} membeli {cw} {unit} {item} seharga {pw} rupiah untuk setiap {unit}.",
        f"Pesanan {p} berupa {cw} {unit} {item} akan dikirim dari {city}; harganya {pw} rupiah per {unit}."])
    qa = [(f"Berapa {unit} {item} yang dipesan {p}?", cw), (f"Dari kota mana {item} itu dipesan?", city),
          (f"Berapa harga {item} per {unit}?", pw)]
    un = [f"Kapan pesanan {p} tiba?", f"Siapa yang mengirim {item} itu?"]
    return note, qa, un


def fam_cause(rng):
    ev, cause, day = _topic_event(rng), pick(rng, CAUSES), pick(rng, DAYS)
    note = rng.choice([
        f"{ev.capitalize()} ditunda sampai hari {day} karena {cause}.",
        f"Akibat {cause}, {ev} yang semula direncanakan hari ini dialihkan ke hari {day}.",
        f"Panitia menunda {ev} hingga {day} sebab terjadi {cause}."])
    qa = [(rng.choice([f"Mengapa {ev} ditunda?", f"Apa penyebab {ev} ditunda?"]), cause),
          (f"Sampai hari apa {ev} ditunda?", day)]
    un = [f"Siapa yang memutuskan penundaan {ev}?", f"Di mana {ev} akan diadakan?"]
    return note, qa, un


def fam_role(rng):
    a, b = G.person_name(rng), G.person_name(rng)
    role, year = pick(rng, ROLES), str(rng.randint(2005, 2024))
    note = rng.choice([
        f"{a} menjabat sebagai {role} sejak tahun {year}, menggantikan {b} yang telah pensiun.",
        f"Sejak {year}, jabatan {role} dipegang oleh {a}, setelah {b} mengundurkan diri.",
        f"{b} sebelumnya menjadi {role}; kini posisi itu diisi {a} sejak {year}."])
    qa = [(f"Siapa yang menjabat sebagai {role}?", a), (f"Siapa yang digantikan {a}?", b),
          (f"Sejak kapan {a} menjabat?", year)]
    un = [f"Berapa gaji {a}?", f"Di mana kantor {a}?"]
    return note, qa, un


def fam_siblings(rng):
    s, a, b = G.person_name(rng), G.person_name(rng).split()[0], G.person_name(rng).split()[0]
    note = rng.choice([
        f"Saudara tertua {s} bernama {a}, sedangkan yang paling muda bernama {b}.",
        f"{s} punya seorang kakak bernama {a} dan seorang adik bernama {b}.",
        f"Kakak {s} adalah {a}, adiknya {b}."])
    qa = [(f"Siapa nama kakak {s}?", a), (f"Siapa nama adik {s}?", b)]
    un = [f"Siapa nama ayah {s}?", f"Di mana {s} bersekolah?"]
    return note, qa, un


def fam_allergy(rng):
    s = G.person_name(rng)
    x, y = rng.sample(ALLERGY, 2)
    note = rng.choice([
        f"{s} alergi terhadap {x} dan {y}, sehingga harus menghindari hidangan itu.",
        f"Dokter menyarankan {s} tidak makan {x} maupun {y} karena alergi.",
        f"Daftar pantangan {s}: {x} dan {y}."])
    qa = [(f"Apa saja yang membuat {s} alergi?", f"{x} dan {y}"), (f"{s} tidak boleh makan apa?", f"{x} dan {y}")]
    un = [f"Sejak kapan {s} alergi?", f"Siapa dokter {s}?"]
    return note, qa, un


def fam_storage(rng):
    s, obj, cont, place = G.person_name(rng), pick(rng, ["resep warisan", "buku catatan", "surat penting", "kunci cadangan", "foto lama"]), pick(rng, CONTAINERS), pick(rng, ROOMS)
    note = rng.choice([
        f"{s} menyimpan {obj} di dalam {cont} yang ada di ruang {place}.",
        f"{obj.capitalize()} milik {s} diletakkan di {cont}, tepatnya di ruang {place}.",
        f"Kalau mencari {obj} {s}, lihatlah di {cont} dalam ruang {place}."])
    qa = [(f"Di mana {s} menyimpan {obj}?", cont), (f"Di ruang apa {obj} {s} disimpan?", place)]
    un = [f"Sejak kapan {obj} itu disimpan?", f"Siapa yang memberikan {obj} kepada {s}?"]
    return note, qa, un


def fam_receipt(rng):
    item, p = pick(rng, ["paket", "surat", "barang kiriman", "kardus"]), _person(rng)
    d1, d2 = rng.sample(DAYS, 2)
    city, place = G.word(rng, 3).capitalize(), pick(rng, PLACES)
    note = rng.choice([
        f"{item.capitalize()} dari {city} yang dikirim hari {d1} baru diterima oleh {p} pada hari {d2} di {place}.",
        f"Pada hari {d2}, {p} menerima {item} kiriman dari {city} di {place}; {item} itu dikirim sejak hari {d1}.",
        f"{p} mengambil {item} dari {city} di {place} hari {d2}, padahal pengiriman dilakukan hari {d1}."])
    qa = [(f"Siapa yang menerima {item} dari {city}?", p), (f"Di mana {item} itu diterima?", place),
          (f"Kapan {item} itu diterima?", d2)]
    un = [f"Berapa isi {item} itu?", f"Siapa pengirim {item} dari {city}?"]
    return note, qa, un


def fam_transport(rng):
    sv, a, b = pick(rng, SERVICES), G.word(rng, 3).capitalize(), G.word(rng, 3).capitalize()
    gap, dur = num_words(rng.randint(1, 4)), num_words(rng.randint(1, 8))
    note = rng.choice([
        f"{sv.capitalize()} dari {a} ke {b} berangkat setiap {gap} jam sekali dengan lama perjalanan sekitar {dur} jam.",
        f"Perjalanan {sv} dari {a} menuju {b} memakan waktu kira-kira {dur} jam, dan keberangkatan terjadi tiap {gap} jam.",
        f"Jadwal {sv} {a}-{b}: berangkat setiap {gap} jam, tiba setelah {dur} jam."])
    qa = [(f"Berapa lama perjalanan {sv} dari {a} ke {b}?", f"{dur} jam"), (f"Seberapa sering {sv} dari {a} berangkat?", f"{gap} jam")]
    un = [f"Berapa harga tiket {sv} dari {a}?", f"Di mana terminal {sv} di {a}?"]
    return note, qa, un


def fam_prices(rng):
    shop = f"{pick(rng, ['Toko', 'Warung', 'Rumah Makan'])} {pick(rng, SHOP)} {pick(rng, ['Jaya', 'Rejeki', 'Abadi', 'Mulia'])}"
    f1, f2 = rng.sample(FOODS2, 2)
    p1, _ = price_words(rng)
    p2, _ = price_words(rng)
    note = rng.choice([
        f"{shop} menjual {f1} seharga {p1} rupiah dan {f2} seharga {p2} rupiah.",
        f"Di {shop}, {f1} dihargai {p1} rupiah, sedangkan {f2} dihargai {p2} rupiah.",
        f"Daftar harga {shop}: {f1} {p1} rupiah, {f2} {p2} rupiah."])
    qa = [(f"Berapa harga {f1} di {shop}?", p1), (f"Berapa harga {f2} di {shop}?", p2)]
    un = [f"Jam berapa {shop} buka?", f"Siapa pemilik {shop}?"]
    return note, qa, un


FAMILIES = [fam_schedule, fam_two_facts, fam_order, fam_cause, fam_role, fam_siblings, fam_allergy,
            fam_storage, fam_receipt, fam_transport, fam_prices]


def make_note(rng):
    return rng.choice(FAMILIES)(rng)


def make_example2(rng):
    """dict(ctx, question, answer|None, kind). Konteks: catatan utama + 0-2 pengecoh (v2 atau v1)."""
    note, qa, un = make_note(rng)
    r = rng.random()
    distractors = []
    for _ in range(rng.choice([0, 0, 1, 2])):
        if rng.random() < 0.6:
            distractors.append(make_note(rng)[0])
        else:
            distractors.append(G.make_fact(rng, rng.choice(G.TRAIN_ATTRS), split="A")["note"])
    if r < 0.18:  # tak terjawab tapi berkaitan dengan catatan yang ada
        q = rng.choice(un)
        notes = [note] + distractors
        rng.shuffle(notes)
        return {"ctx": " ".join(notes), "question": q, "answer": None, "kind": "unanswerable_related"}
    if r < 0.26 and distractors:  # tak terjawab: pertanyaan tentang catatan lain yang tidak ada di konteks
        other_note, other_qa, _ = make_note(rng)
        q = rng.choice(other_qa)[0]
        notes = [note] + distractors
        rng.shuffle(notes)
        return {"ctx": " ".join(notes), "question": q, "answer": None, "kind": "unanswerable_absent"}
    q, a = rng.choice(qa)
    notes = [note] + distractors
    rng.shuffle(notes)
    return {"ctx": " ".join(notes), "question": q, "answer": a, "kind": "answerable"}
