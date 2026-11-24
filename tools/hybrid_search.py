"""
Pencarian hibrida untuk Memori: BM25 (FTS5) + vektor n-gram (numpy saja) digabung dengan Reciprocal Rank Fusion.

Vektor di sini BUKAN embedding neural: TF-IDF kata dan n-gram karakter yang di-hash ke dimensi tetap
(opsional diringkas dengan LSA/SVD agar kata yang sering muncul bersama saling mendekat). Ia membantu untuk
variasi imbuhan/ejaan ("menggunakan" ~ "penggunaan") dan, dengan LSA pada kumpulan besar, sedikit sinonim.
Ia tidak menggantikan embedding terlatih.
"""
import re
import zlib

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = set("""yang di ke dari dan atau apa apakah siapa kapan dimana dimanakah mana manakah bagaimana berapa mengapa kenapa
adalah ialah itu ini untuk dengan pada dalam oleh sebagai kah nya pun akan telah sudah dapat bisa para saja lah tahun
terjadi disebut dinamakan bernama yaitu yakni tersebut dia ia mereka kita kami saya anda ada tidak bukan""".split())


def _h(s):
    return zlib.crc32(s.encode("utf-8"))


def features(text, ngram=(3, 4, 5), word_weight=2.0):
    """Daftar (fitur, bobot): kata utuh dan n-gram karakter dari tiap kata."""
    feats = {}
    for w in _WORD.findall(text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        feats["w:" + w] = feats.get("w:" + w, 0.0) + word_weight
        padded = "<" + w + ">"
        for n in ngram:
            if len(padded) < n:
                continue
            for i in range(len(padded) - n + 1):
                g = "c:" + padded[i:i + n]
                feats[g] = feats.get(g, 0.0) + 1.0
    return feats


class VectorIndex:
    """Indeks vektor padat kecil. ids[i] cocok dengan baris ke-i matriks."""

    def __init__(self, dim=2048, lsa_dim=0, seed=0):
        self.dim = int(dim)
        self.lsa_dim = int(lsa_dim)
        self.seed = seed
        self.ids = []
        self.matrix = np.zeros((0, self.dim), dtype=np.float32)
        self.idf = None
        self.proj = None   # (dim, lsa_dim) bila LSA aktif

    def _hash_vec(self, feats):
        v = np.zeros(self.dim, dtype=np.float32)
        for f, wt in feats.items():
            h = _h(f)
            v[h % self.dim] += wt * (1.0 if (h >> 20) & 1 else -1.0) * (self.idf[h % self.dim] if self.idf is not None else 1.0)
        return v

    def fit(self, ids, texts):
        self.ids = list(ids)
        feats = [features(t) for t in texts]
        df = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            for k in {_h(x) % self.dim for x in f}:
                df[k] += 1.0
        n = max(1, len(feats))
        self.idf = (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        m = np.zeros((len(feats), self.dim), dtype=np.float32)
        for i, f in enumerate(feats):
            m[i] = self._hash_vec(f)
        self.proj = None
        k = min(self.lsa_dim, len(feats) - 1, self.dim) if self.lsa_dim else 0
        if k >= 8:
            # LSA: ambil k arah utama dari kumpulan teks ini (numpy.linalg.svd, tanpa scipy)
            _, _, vt = np.linalg.svd(m - 0.0, full_matrices=False)
            self.proj = vt[:k].T.astype(np.float32)
            m = m @ self.proj
        self.matrix = _unit(m)
        return self

    def encode(self, text):
        v = self._hash_vec(features(text))
        if self.proj is not None:
            v = v @ self.proj
        return _unit(v[None, :])[0]

    def search(self, query, k=50):
        if not len(self.ids):
            return []
        q = self.encode(query)
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order if sims[i] > 0]


def _unit(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


def rrf(rankings, k=60, weights=None):
    """Reciprocal Rank Fusion. rankings: daftar daftar id (terbaik dulu). Kembalikan id urut skor menurun."""
    score = {}
    for r, ranking in enumerate(rankings):
        wt = 1.0 if weights is None else weights[r]
        for pos, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + wt / (k + pos + 1)
    return sorted(score, key=lambda i: (-score[i], str(i)))


# ---------------------------------------------------------------------------------------------------------------------
# Pencarian hibrida untuk MemoryStore: BM25 + vektor embedding terlatih, digabung RRF. Bawaan MATI (MEMORI_HYBRID=on).
# Hasil ukur (tools/hybrid_eval.py, README): vektor n-gram buatan numpy TIDAK membantu; embedding terlatih membantu pada
# paragraf panjang. Karena itu lapisan ini memakai embedder yang disuntikkan, bukan vektor n-gram.
# ---------------------------------------------------------------------------------------------------------------------
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_MIN_SIM = 0.86   # ukur: pertanyaan di luar topik 0/8 lolos; paragraf benar 77,8% lolos (lihat README)


class E5Embedder:
    """Pembungkus embedding gaya E5 (awalan 'query: ' / 'passage: ', mean pooling). Butuh torch + transformers (opsional)."""

    def __init__(self, name=DEFAULT_EMBED_MODEL, max_length=256, batch_size=32):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.name = name
        self.max_length = max_length
        self.batch_size = batch_size
        self._torch = torch
        try:   # utamakan cache lokal: tanpa menghubungi huggingface.co (offline tidak menunggu percobaan ulang puluhan detik)
            self._tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            self._mdl = AutoModel.from_pretrained(name, local_files_only=True).eval()
        except Exception:   # belum ada di cache: unduh sekali
            self._tok = AutoTokenizer.from_pretrained(name)
            self._mdl = AutoModel.from_pretrained(name).eval()

    def _encode(self, texts, prefix):
        torch = self._torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            x = self._tok([prefix + t for t in texts[i:i + self.batch_size]], padding=True, truncation=True,
                          max_length=self.max_length, return_tensors="pt")
            with torch.no_grad():
                h = self._mdl(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 1), dtype=np.float32)

    def encode_passages(self, texts):
        return self._encode(list(texts), "passage: ")

    def encode_query(self, text):
        return self._encode([text], "query: ")[0]


class NoteVectors:
    """Vektor catatan disimpan di tabel note_vec (SQLite yang sama) sehingga tidak dihitung ulang. Hanya catatan baru yang dikodekan."""

    def __init__(self, con, embedder):
        self.con, self.embedder = con, embedder
        con.execute("CREATE TABLE IF NOT EXISTS note_vec(note_id INTEGER PRIMARY KEY, model TEXT, vec BLOB)")
        self._cache = None   # (kunci, ids, matriks)

    def _sync(self):
        name = self.embedder.name
        self.con.execute("DELETE FROM note_vec WHERE model<>?", (name,))
        self.con.execute("DELETE FROM note_vec WHERE note_id NOT IN (SELECT id FROM notes WHERE active=1)")
        todo = self.con.execute("SELECT id, text FROM notes WHERE active=1 AND id NOT IN (SELECT note_id FROM note_vec)").fetchall()
        if todo:
            vecs = self.embedder.encode_passages([t for _, t in todo])
            self.con.executemany("INSERT OR REPLACE INTO note_vec(note_id, model, vec) VALUES (?,?,?)",
                                 [(i, name, v.tobytes()) for (i, _), v in zip(todo, vecs)])
        self.con.commit()

    def matrix(self):
        self._sync()
        key = self.con.execute("SELECT COUNT(*), COALESCE(MAX(note_id),0), COALESCE(SUM(note_id),0) FROM note_vec").fetchone()
        if self._cache is None or self._cache[0] != key:
            rows = self.con.execute("SELECT note_id, vec FROM note_vec ORDER BY note_id").fetchall()
            ids = [r[0] for r in rows]
            mat = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1), dtype=np.float32)
            self._cache = (key, ids, mat)
        return self._cache[1], self._cache[2]

    def search(self, question, k=30, allowed=None, min_sim=0.0):
        """id terdekat dengan kemiripan >= min_sim. Tanpa batas bawah, vektor SELALU mengembalikan 'yang terdekat' walau tak relevan
        (embedding E5 sangat rapat: paragraf benar median 0,887, 'terdekat tapi salah' median 0,839)."""
        ids, mat = self.matrix()
        if not ids:
            return []
        sims = mat @ self.embedder.encode_query(question)
        out = []
        for j in np.argsort(-sims):
            if sims[j] < min_sim:
                break
            if allowed is None or ids[j] in allowed:
                out.append(ids[j])
                if len(out) >= k:
                    break
        return out


def fuse(bm25_ids, dense_ids, k):
    """RRF dengan bobot sama (hasil ukur: bobot lain tidak lebih baik); kembalikan k id teratas."""
    return rrf([bm25_ids, dense_ids])[:k]
