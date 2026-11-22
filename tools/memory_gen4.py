"""
Pembangkit SFT tahap 7 (v7). Menambah pada memory_gen3 hal-hal yang terbukti kurang pada hasil v6:
  - pertanyaan BALIK pada dua kalimat kembar (tim apa yang rapat di ruang Mawar), dengan nama tim kata nyata (finance, produk, ...)
  - dua orang dengan dua mata pelajaran atau pekerjaan, ditanya dari kedua arah, jawaban memakai gelar seperti di catatan (Pak Anwar)
  - jawaban frasa tempat lengkap (pantai Parangtritis, kantor pos Cimahi), karena v6 memotong jawaban menjadi kantor pos atau brankas
  - dua harga dalam kata (dua puluh lima ribu lawan lima ribu), ditanya dari kedua arah
  - rentang tanggal dalam kata (mulai ... sampai ...), karena v6 menjawab titik awal untuk pertanyaan titik akhir
Set uji (evalsets/*, data/sft/memory_complex*, memory_handwritten*) TIDAK dipakai dan tidak boleh disalin ke sini; tidak ada kalimat uji yang disalin.
"""
import memory_gen as G
import memory_gen2 as G2
import memory_gen3 as G3

TEAM_WORDS = ["pemasaran", "logistik", "keuangan", "desain", "gudang", "hukum", "teknologi", "layanan", "pengadaan", "riset", "keamanan", "humas", "mutu"]
EVENTS = ["Rapat", "Pelatihan", "Pertemuan", "Evaluasi", "Latihan", "Presentasi"]
SUBJECTS = ["matematika", "fisika", "kimia", "biologi", "sejarah", "geografi", "bahasa Inggris", "seni musik", "olahraga", "ekonomi"]
JOBS = ["koki", "sopir", "montir", "penjahit", "tukang kebun", "petugas kebersihan", "kasir", "penjaga gudang"]
PLACE_TYPES = ["pantai", "kantor pos", "taman", "terminal", "balai desa", "stasiun", "pasar", "bank cabang", "gedung serbaguna", "lapangan"]
GOODS = ["nasi campur", "mie ayam", "es teh", "soto", "bakso", "kopi susu", "roti bakar", "jus jeruk"]
HELD_OUT_ROOMS = {"Mawar", "Anggrek"}
TITLES = ["Bu", "Pak", "Ibu", "Bapak"]
DATE_MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


def _place(rng):
    return f"{rng.choice(PLACE_TYPES)} {G.word(rng, 3).capitalize()}"


def _titled(rng):
    return f"{rng.choice(TITLES)} {G.word(rng, 2).capitalize()}"


def fam_twin_events(rng):
    ev = rng.choice(EVENTS)
    a, b = rng.sample(TEAM_WORDS, 2)
    ta, tb = G3._clock(rng, 7, 10), G3._clock(rng, 11, 16)
    ra, rb = rng.sample([r for r in G2.ROOMS if r not in HELD_OUT_ROOMS], 2)      # ruang Mawar/Anggrek dipakai kasus uji nyata: tidak dilatih
    verb = rng.choice(["dimulai pukul", "mulai pukul", "berlangsung pukul"])
    n1 = f"{ev} tim {a} {verb} {ta} di ruang {ra}."
    n2 = f"{ev} tim {b} {verb} {tb} di ruang {rb}."
    note = f"{n1} {n2}" if rng.random() < 0.5 else f"{n2} {n1}"
    ev_l = ev.lower()
    qa = [(f"tim apa yang {ev_l} di ruang {rb}", b), (f"tim mana yang {ev_l} di ruang {ra}", a), (f"{ev_l} di ruang {rb} diadakan oleh tim mana", b),
          (f"tim apa yang mulai {ev_l} pukul {ta}", a), (f"jam berapa {ev_l} tim {b}", tb), (f"di ruang mana {ev_l} tim {a}", ra),
          (f"Pukul berapa {ev_l} tim {a} dimulai?", ta)]
    un = [f"siapa ketua tim {a}", f"apa yang dibahas dalam {ev_l} tim {b}"]
    return note, qa, un


def fam_twin_people(rng):
    p1, p2 = _titled(rng), _titled(rng)
    while p1 == p2:
        p2 = _titled(rng)
    if rng.random() < 0.5:
        s1, s2 = rng.sample(SUBJECTS, 2)
        k1, k2 = G2.num_words(rng.randint(3, 9)), G2.num_words(rng.randint(3, 9))
        note = rng.choice([
            f"{p1} mengajar {s1} di kelas {k1}, sementara {p2} mengajar {s2} di kelas {k2}.",
            f"{p2} mengajar {s2} di kelas {k2}, sedangkan {p1} mengajar {s1} di kelas {k1}."])
        qa = [(f"Siapa yang mengajar {s1}?", p1), (f"Siapa yang mengajar {s2}?", p2), (f"{p1} mengajar apa?", s1), (f"{p2} mengajar apa?", s2)]
        un = [f"Siapa yang mengajar {rng.choice([x for x in SUBJECTS if x not in (s1, s2)])}?", f"Di mana {p1} tinggal?"]
    else:
        j1, j2 = rng.sample(JOBS, 2)
        note = rng.choice([f"{p1} bekerja sebagai {j1}, dan {p2} bekerja sebagai {j2}.", f"Pekerjaan {p1} adalah {j1}, sedangkan {p2} menjadi {j2}."])
        qa = [(f"Siapa yang bekerja sebagai {j1}?", p1), (f"Siapa yang bekerja sebagai {j2}?", p2), (f"Apa pekerjaan {p1}?", j1), (f"Apa pekerjaan {p2}?", j2)]
        un = [f"Siapa yang bekerja sebagai {rng.choice([x for x in JOBS if x not in (j1, j2)])}?", f"Berapa gaji {p2}?"]
    return note, qa, un


def fam_place_phrase(rng):
    pl, pl2 = _place(rng), _place(rng)
    thing, who = rng.choice(["Paket", "Surat", "Barang kiriman", "Dokumen"]), G.person_name(rng)
    day = rng.choice(G2.DAYS)
    kind = rng.randint(0, 2)
    if kind == 0:
        note = f"{thing} dari {G.word(rng, 3).capitalize()} baru diterima oleh {who} pada hari {day} di {pl}."
        qa = [(f"Di mana {thing.lower()} itu diterima?", pl), (f"Siapa yang menerima {thing.lower()} itu?", who), (f"Kapan {thing.lower()} itu diterima?", day)]
    elif kind == 1:
        note = f"Lomba lari akan dimulai dari {pl} dan berakhir di {pl2}."
        qa = [("Di mana lomba lari berakhir?", pl2), ("Dari mana lomba lari dimulai?", pl)]
    else:
        note = f"Festival tahunan diadakan di {pl} pada hari {day}, sedangkan penutupannya di {pl2}."
        qa = [("Di mana festival tahunan diadakan?", pl), ("Di mana penutupan festival?", pl2), ("Kapan festival tahunan diadakan?", day)]
    un = ["Berapa biaya masuknya?", f"Siapa yang mengantar {thing.lower()} itu?"]
    return note, qa, un


def fam_twin_prices(rng):
    g1, g2 = rng.sample(GOODS, 2)
    p1, k1 = G2.price_words(rng)
    p2, k2 = G2.price_words(rng)
    while k1 == k2:
        p2, k2 = G2.price_words(rng)
    shop = f"{rng.choice(['Warung', 'Rumah Makan', 'Kedai'])} {rng.choice(G2.SHOP)} {rng.choice(['Jaya', 'Rejeki', 'Mulia'])}"
    note = rng.choice([
        f"{shop} menyediakan menu {g1} seharga {p1} dan {g2} seharga {p2} rupiah.",
        f"Di {shop}, {g1} dihargai {p1} sedangkan {g2} dihargai {p2} rupiah."])
    qa = [(f"Berapa harga {g1}?", p1), (f"Berapa harga {g2}?", p2), (f"Apa yang harganya {p2}?", g2), (f"Menu apa yang seharga {p1}?", g1)]
    un = [f"Jam berapa {shop} buka?", f"Berapa harga {rng.choice([x for x in GOODS if x not in (g1, g2)])}?"]
    return note, qa, un


def _date_words(rng):
    return f"{G2.num_words(rng.randint(1, 28))} {rng.choice(DATE_MONTHS)}"


def fam_period(rng):
    d1, d2 = _date_words(rng), _date_words(rng)
    while d1 == d2:
        d2 = _date_words(rng)
    what = rng.choice(["Sekolah libur", "Kantor tutup", "Pendaftaran dibuka", "Kursus berlangsung", "Pameran diadakan"])
    note = rng.choice([f"{what} mulai tanggal {d1} sampai dengan tanggal {d2}.", f"{what} dari {d1} hingga {d2}."])
    key = what.split()[0].lower()
    qa = [(f"Sampai kapan {what.lower()}?", d2), (f"Sejak kapan {what.lower()}?", d1), (f"Kapan {what.lower()} dimulai?", d1), (f"Kapan {what.lower()} berakhir?", d2)]
    un = [f"Siapa yang mengumumkan {key}?", f"Di mana {what.lower()}?"]
    return note, qa, un


NEW_FAMILIES = [fam_twin_events, fam_twin_people, fam_place_phrase, fam_twin_prices, fam_period]
ALL_FAMILIES = NEW_FAMILIES + G3.FAMILIES3


def make_note4(rng):
    return rng.choice(ALL_FAMILIES)(rng)


def make_example4(rng, p_new=0.65):
    """p_new: peluang famili baru (v7) dibanding famili tahap 6. Pengecoh sering satu famili yang sama dengan nilai lain."""
    fam = rng.choice(NEW_FAMILIES) if rng.random() < p_new else rng.choice(G3.FAMILIES3)
    note, qa, un = fam(rng)
    distractors = []
    for _ in range(rng.choice([0, 0, 1, 2])):
        r = rng.random()
        if r < 0.35:
            distractors.append(fam(rng)[0])
        elif r < 0.65:
            distractors.append(make_note4(rng)[0])
        elif r < 0.9:
            distractors.append(G2.make_note(rng)[0])
        else:
            distractors.append(G.make_fact(rng, rng.choice(G.TRAIN_ATTRS), split="A")["note"])
    notes = [note] + distractors
    r = rng.random()
    if r < 0.15:
        rng.shuffle(notes)
        return {"ctx": " ".join(notes), "question": rng.choice(un), "answer": None, "kind": "unanswerable_related"}
    if r < 0.22 and distractors:
        ctx = " ".join(notes)
        other_qa = [(q, x) for q, x in fam(rng)[1] if x not in ctx]      # jawabannya tidak boleh kebetulan ada di konteks
        if other_qa:
            rng.shuffle(notes)
            return {"ctx": " ".join(notes), "question": rng.choice(other_qa)[0], "answer": None, "kind": "unanswerable_absent"}
    q, a = rng.choice(qa)
    rng.shuffle(notes)
    return {"ctx": " ".join(notes), "question": q, "answer": a, "kind": "answerable"}
