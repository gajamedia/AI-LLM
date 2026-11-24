"""Tes pencarian hibrida (BM25 + vektor): RRF, penyimpan vektor, integrasi MemoryStore dengan embedder palsu (tanpa unduhan)."""
import numpy as np

import hybrid_search as H
import memory_agent as M


class FakeEmbedder:
    """Embedder palsu: sinonim ('mobil' ~ 'kendaraan') memiliki sumbu yang sama, sehingga BM25 murni tidak menemukannya."""
    name = "fake-v1"

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        t = text.lower()
        v = np.zeros(8, dtype=np.float32)
        v[0] = 1.0 if ("mobil" in t or "kendaraan" in t) else 0.0
        v[1] = 1.0 if "kopi" in t else 0.0
        v[2] = 1.0 if "rapat" in t else 0.0
        v[4 + sum(map(ord, t)) % 4] = 0.01 if v[:3].any() else 1.0    # teks tanpa kata kunci: arah acak per teks, bukan satu arah yang sama
        return v / np.linalg.norm(v)

    def encode_passages(self, texts):
        self.calls += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, 8), dtype=np.float32)

    def encode_query(self, text):
        return self._vec(text)


def _store(embedder=None):
    s = M.MemoryStore(":memory:")
    s.teach("Mobil dinas dipakai untuk antar tamu ke bandara.")
    s.teach("Mesin kopi di dapur kantor bermerek Latte.")
    s.teach("Rapat tim produk di ruang Anggrek.")
    if embedder is not None:
        s._hv_tried = True
        s._hv = H.NoteVectors(s.con, embedder)
    return s


def test_rrf_and_fuse():
    assert H.rrf([[1, 2, 3], [3, 2, 1]]) == [1, 3, 2]     # 1/61+1/63 sedikit di atas 2/62
    assert H.fuse([1, 2], [9, 1], 2)[0] == 1               # id 1 ada di kedua daftar
    assert H.rrf([[]]) == []


def test_default_off_is_plain_bm25():
    s = _store()
    assert s._hybrid_vectors() is None
    assert [r["id"] for r in s.search("kendaraan apa", k=3)] == []      # BM25 murni tidak tahu sinonim
    assert s.search("kopi bermerek apa", k=1)[0]["id"] == 2


def test_hybrid_finds_synonym_note():
    s = _store(FakeEmbedder())
    got = s.search("kendaraan apa", k=3)
    assert got and got[0]["id"] == 1                       # ketemu lewat vektor walau tidak ada kata yang sama
    assert s.search("kopi bermerek apa", k=1)[0]["id"] == 2   # hasil yang BM25 benar tetap benar


def test_vectors_cached_and_follow_forget_and_new_notes():
    emb = FakeEmbedder()
    s = _store(emb)
    s.search("kopi", k=1)
    first = emb.calls
    assert first == 3
    s.search("kopi", k=1)
    assert emb.calls == first                              # tidak dikodekan ulang
    nid = s.teach("Kendaraan operasional baru tiba.")
    got = [r["id"] for r in s.search("kendaraan", k=5)]
    assert emb.calls == first + 1 and nid in got
    assert s.forget(nid)
    assert nid not in [r["id"] for r in s.search("kendaraan", k=5)]


def test_model_change_reencodes():
    emb = FakeEmbedder()
    s = _store(emb)
    s.search("kopi", k=1)
    emb.name = "fake-v2"
    s.search("kopi", k=1)
    assert emb.calls == 6


def test_docs_only_filters_dense_results():
    emb = FakeEmbedder()
    s = _store(emb)
    s.add_document("armada.md", ["Armada kendaraan perusahaan terdiri dari tiga unit."])
    got = s.search("kendaraan", k=5, docs_only=True)
    assert got and all(r["doc_id"] is not None for r in got)


def test_failure_falls_back_to_bm25():
    class Broken(FakeEmbedder):
        def encode_query(self, text):
            raise RuntimeError("gagal")
    s = _store(Broken())
    assert s.search("kopi bermerek apa", k=1)[0]["id"] == 2


def test_setting_on_without_deps_falls_back(monkeypatch):
    monkeypatch.setenv("MEMORI_HYBRID", "on")
    monkeypatch.setenv("MEMORI_EMBED_MODEL", "tidak/ada-model-ini")
    s = M.MemoryStore(":memory:")
    s.teach("Mesin kopi bermerek Latte.")
    assert s.search("kopi", k=1)[0]["id"] == 1             # gagal memuat embedder -> BM25, tidak error


def test_ngram_vector_index_basics():
    ids = [0, 1, 2]
    texts = ["penggunaan mesin kopi di dapur", "rapat tim produk hari kamis", "jadwal penerbangan ke bandara"]
    vi = H.VectorIndex(dim=512).fit(ids, texts)
    assert vi.search("menggunakan mesin kopi")[0][0] == 0


def test_dense_has_similarity_floor_so_irrelevant_query_returns_nothing():
    s = _store(FakeEmbedder())
    assert s.search("tujuan artikel", k=3) == []                        # tak ada kata sama dan vektor tak mirip: jangan menampilkan "yang terdekat"
    s.add_document("artikel.txt", ["Tujuan artikel ada dua sudut pandang."])
    assert s.search("pertanyaan lain sama sekali", k=3, docs_only=True) == []


def test_min_sim_setting_can_be_lowered(monkeypatch):
    s = _store(FakeEmbedder())
    monkeypatch.setenv("MEMORI_HYBRID_MIN_SIM", "-1")                   # tanpa batas: kembali ke perilaku 'selalu ada yang terdekat'
    assert s.search("tujuan artikel", k=3)
