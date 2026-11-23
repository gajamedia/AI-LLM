import memory_agent as M

N1 = "Rapat tim produk dimulai pukul 09.30 di ruang Anggrek."
N2 = "Rapat tim finance dimulai pukul 11.30 di ruang Mawar."


def _lk(q, notes):
    return M.category_lookup(q, [(i, n, "") for i, n in enumerate(notes, 1)])


def test_menjawab_kata_setelah_kategori_untuk_berbagai_susunan():
    for q in ["tim mana yang menggunakan ruangan anggrek?", "yang menggunakan ruangan anggrek itu tim mana", "ruangan anggrek digunakan oleh tim apa?",
              "tim apa yang rapat di ruang anggrek", "rapat di ruang anggrek diadakan oleh tim mana"]:
        assert _lk(q, [N1, N2])["answer"] == "produk", q
        assert _lk(q.replace("anggrek", "mawar"), [N1, N2])["answer"] == "finance", q


def test_berhenti_bila_tidak_yakin():
    assert _lk("tim mana yang rapat di ruang melati", [N1, N2]) is None                 # ruang tidak ada di catatan mana pun
    assert _lk("tim mana yang rapat di ruang melati", [N1]) is None                    # satu catatan: kata acuan 'melati' tak ada, bukan kata kerja
    assert _lk("tim mana yang rapat di ruang anggrek", ["Rapat tim produk dan tim finance di ruang Anggrek."]) is None   # dua jawaban berbeda
    assert _lk("jam berapa rapat di ruang anggrek", [N1, N2]) is None                    # bukan pola 'kategori mana/apa'
    assert _lk("di mana rapat tim produk", [N1, N2]) is None
    assert _lk("tim mana yang rapat", [N1, N2]) is None                                 # tak ada kata pembeda antar catatan
    assert _lk("kota mana yang dikunjungi Bu Rina", ["Bu Rina tinggal di Jalan Mawar nomor 5."]) is None


def test_kategori_lain_dan_rangkaian_nama():
    assert _lk("kota apa yang dikunjungi Bu Rina", ["Bu Rina mengunjungi kota Surabaya Timur kemarin."])["answer"] == "Surabaya Timur"
    assert _lk("hari apa rapat RT", ["Rapat RT diadakan hari Sabtu pukul 19.00."])["answer"] == "Sabtu"
    assert _lk("tim mana yang di ruang anggrek", ["Tim yang rapat di ruang Anggrek adalah produk."]) is None   # kata setelah 'tim' adalah kata penghubung


def test_agent_hanya_memakai_cadangan_saat_model_tidak_menjawab():
    class Echo:
        def read(self, ctx, q):
            return "Anggrek" if "Anggrek" in ctx else "Mawar"

    class Right:
        def read(self, ctx, q):
            return "produk" if "Anggrek" in ctx else "finance"

    store = M.MemoryStore(":memory:")
    store.teach(N1)
    store.teach(N2)
    a = M.MemoryAgent(store, Echo())
    o = a.ask("tim mana yang menggunakan ruangan anggrek?")
    assert o["status"] == "ok" and o["answer"] == "produk" and o["via"] == "category_lookup" and o["source"]["text"] == N1
    a.category_fallback = False
    assert a.ask("tim mana yang menggunakan ruangan anggrek?")["status"] == "unknown"
    b = M.MemoryAgent(store, Right())
    ob = b.ask("tim mana yang rapat di ruang anggrek")
    assert ob["answer"] == "produk" and "via" not in ob                              # jawaban model dipakai; cadangan tidak berjalan
