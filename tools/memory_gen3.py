"""
Pembangkit catatan jenis baru untuk SFT tahap 6: tim, divisi, peran, jam buka dan tutup, anggota, ruang, mulai dan selesai.
Format sama dengan memory_gen2 (famili mengembalikan note, qa, un). Setiap jawaban adalah potongan persis dari catatan (pembaca ekstraktif).
Sengaja banyak pasangan yang mudah tertukar (buka lawan tutup, ketua lawan wakil, mulai lawan selesai, divisi lawan tim) dan catatan
dengan dua nilai sejenis dalam satu kalimat, karena di situlah pembaca 100M sering salah.
Set uji (evalsets/memory_*.jsonl, data/sft/memory_complex*.jsonl) TIDAK dipakai dan tidak boleh disalin ke sini.
"""
import memory_gen as G
import memory_gen2 as G2

DIVISIONS = ["Keuangan", "Pemasaran", "Produksi", "Logistik", "Pengadaan", "Humas", "Teknologi", "Layanan Pelanggan", "Pengawasan Mutu", "Sumber Daya Manusia"]
TEAM_ROLES = ["ketua", "wakil ketua", "sekretaris", "bendahara", "koordinator lapangan", "penanggung jawab", "juru bicara"]
SHOPS = ["perpustakaan", "koperasi", "klinik", "kantin", "bengkel", "toko alat tulis", "apotek", "kantor pos pembantu", "gudang"]
STAGES = ["pendaftaran", "pelatihan", "pembangunan", "penilaian", "pengiriman", "penyuluhan", "pemeriksaan", "pembongkaran"]
HOURS = [f"{h:02d}.00" for h in range(6, 21)]


def _cap(x):
    return x[:1].upper() + x[1:]          # huruf pertama saja (str.capitalize juga mengecilkan nama di dalamnya)


def _clock(rng, lo, hi):
    return f"{rng.randint(lo, hi):02d}.{rng.choice(['00', '30'])}"


def _team(rng):
    return "tim " + G._team(rng)


def fam_team_division(rng):
    t, d, p = _team(rng), rng.choice(DIVISIONS), G.person_name(rng)
    note = rng.choice([
        f"{_cap(t)} berada di bawah divisi {d} dan dipimpin oleh {p}.",
        f"{p} memimpin {t} yang termasuk divisi {d}.",
        f"Divisi {d} membawahi {t}; ketuanya adalah {p}."])
    qa = [(f"{_cap(t)} termasuk divisi apa?", d), (f"Siapa yang memimpin {t}?", p), (f"Divisi apa yang membawahi {t}?", d)]
    un = [f"Berapa anggota {t}?", f"Di ruang mana {t} bekerja?"]
    return note, qa, un


def fam_team_roles(rng):
    t = _team(rng)
    r1, r2 = rng.sample(TEAM_ROLES, 2)
    a, b = G.person_name(rng), G.person_name(rng)
    note = rng.choice([
        f"Di {t}, {a} menjadi {r1} dan {b} menjadi {r2}.",
        f"{_cap(t)} dipegang {a} sebagai {r1}, sedangkan {r2}-nya adalah {b}.",
        f"Susunan {t}: {r1} {a}, {r2} {b}."])
    qa = [(f"Siapa {r1} di {t}?", a), (f"Siapa {r2} di {t}?", b), (f"Apa jabatan {b} di {t}?", r2)]
    un = [f"Siapa yang menjadi {rng.choice([x for x in TEAM_ROLES if x not in (r1, r2)])} di {t}?", f"Kapan {t} dibentuk?"]
    return note, qa, un


def fam_hours(rng):
    shop, name = rng.choice(SHOPS), G2.pick(rng, G2.SHOP)
    open_, close = _clock(rng, 6, 10), _clock(rng, 15, 21)
    full = f"{shop} {name}"
    note = rng.choice([
        f"{_cap(full)} buka pukul {open_} dan tutup pukul {close}.",
        f"Jam tutup {full} adalah {close}, sedangkan jam bukanya {open_}.",
        f"Pelanggan bisa datang ke {full} mulai pukul {open_} sampai pukul {close}."])
    qa = [(f"Pukul berapa {full} buka?", open_), (f"Pukul berapa {full} tutup?", close), (f"Jam berapa {full} mulai melayani?", open_),
          (f"Sampai jam berapa {full} melayani?", close)]
    un = [f"Siapa pemilik {full}?", f"Di jalan mana {full} berada?"]
    return note, qa, un


def fam_start_end(rng):
    stage, proj = rng.choice(STAGES), G._proj(rng)
    d1, d2 = G._date(rng), G._date(rng)
    while d1 == d2:
        d2 = G._date(rng)
    note = rng.choice([
        f"{_cap(stage)} {proj} dimulai pada {d1} dan selesai pada {d2}.",
        f"Untuk {proj}, tahap {stage} berakhir {d2}, setelah dimulai {d1}.",
        f"Mulai {d1} hingga {d2}, {proj} menjalani tahap {stage}."])
    qa = [(f"Kapan tahap {stage} {proj} dimulai?", d1), (f"Kapan tahap {stage} {proj} selesai?", d2)]
    un = [f"Siapa penanggung jawab {proj}?", f"Berapa biaya {stage} {proj}?"]
    return note, qa, un


def fam_member_count(rng):
    t, d = _team(rng), rng.choice(DIVISIONS)
    n_members, n_leads = rng.randint(4, 40), rng.randint(2, 3)
    note = rng.choice([
        f"{_cap(t)} di divisi {d} beranggotakan {n_members} orang, termasuk {n_leads} pengawas.",
        f"Ada {n_members} orang di {t}, divisi {d}, dan {n_leads} di antaranya menjadi pengawas."])
    qa = [(f"Berapa orang anggota {t}?", str(n_members)), (f"Berapa pengawas di {t}?", str(n_leads)), (f"{_cap(t)} ada di divisi apa?", d)]
    un = [f"Siapa ketua {t}?", f"Kapan {t} berkumpul?"]
    return note, qa, un


def fam_room_floor(rng):
    t, room, floor = _team(rng), G2.pick(rng, G2.ROOMS), rng.randint(2, 15)
    person = G.person_name(rng)
    note = rng.choice([
        f"{_cap(t)} bekerja di ruang {room} lantai {floor}; kontaknya {person}.",
        f"Kantor {t} ada di lantai {floor}, ruang {room}, dan {person} yang bisa dihubungi.",
        f"Hubungi {person} untuk urusan {t}, yang berkantor di ruang {room} lantai {floor}."])
    qa = [(f"Di ruang mana {t} bekerja?", room), (f"Di lantai berapa {t} berkantor?", str(floor)), (f"Siapa kontak {t}?", person)]
    un = [f"Berapa nomor telepon {t}?", f"Divisi apa yang membawahi {t}?"]
    return note, qa, un


def fam_deputy(rng):
    t, a, b, c = _team(rng), G.person_name(rng), G.person_name(rng), G.person_name(rng)
    note = rng.choice([
        f"{a} ketua {t}, {b} wakilnya, dan {c} sekretarisnya.",
        f"Ketua {t} adalah {a}. Wakil ketua dijabat {b}, sedangkan sekretaris dijabat {c}."])
    qa = [(f"Siapa ketua {t}?", a), (f"Siapa wakil ketua {t}?", b), (f"Siapa sekretaris {t}?", c)]
    un = [f"Siapa bendahara {t}?", f"Berapa anggota {t}?"]
    return note, qa, un


FAMILIES3 = [fam_team_division, fam_team_roles, fam_hours, fam_start_end, fam_member_count, fam_room_floor, fam_deputy]


def make_note3(rng):
    return rng.choice(FAMILIES3)(rng)


def make_example3(rng, p_new=0.75):
    """Seperti make_example2, tetapi catatan utama dan sebagian pengecoh dari famili baru; pengecoh sering satu famili yang sama (mudah tertukar)."""
    fam = rng.choice(FAMILIES3)
    note, qa, un = fam(rng)
    distractors = []
    for _ in range(rng.choice([0, 0, 1, 2])):
        r = rng.random()
        if r < 0.4:
            distractors.append(fam(rng)[0])                                  # famili sama, nilai lain
        elif r < 0.7:
            distractors.append(make_note3(rng)[0])
        elif r < 0.9:
            distractors.append(G2.make_note(rng)[0])
        else:
            distractors.append(G.make_fact(rng, rng.choice(G.TRAIN_ATTRS), split="A")["note"])
    r = rng.random()
    notes = [note] + distractors
    if r < 0.18:
        rng.shuffle(notes)
        return {"ctx": " ".join(notes), "question": rng.choice(un), "answer": None, "kind": "unanswerable_related"}
    if r < 0.26 and distractors:
        other_note, other_qa, _ = fam(rng)
        rng.shuffle(notes)
        return {"ctx": " ".join(notes), "question": rng.choice(other_qa)[0], "answer": None, "kind": "unanswerable_absent"}
    q, a = rng.choice(qa)
    rng.shuffle(notes)
    return {"ctx": " ".join(notes), "question": q, "answer": a, "kind": "answerable"}
