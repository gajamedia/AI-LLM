"""
Pemilih alat buatan sendiri (tahap 1): regresi logistik multikelas pada n-gram yang di-hash. Hanya numpy; bobotnya kecil.
Rancangan dan alasan: docs/RENCANA_TAHAP1_PEMILIH_ALAT.md. Pelatihan: tools/train_router.py.

Classifier hanya MENGUSULKAN kelas. Pemakainya wajib memverifikasi: kelas alat dipakai hanya bila alat deterministik yang sesuai
menghasilkan jawaban; kelas non-'cari' memerlukan probabilitas >= ambang (decide). Bila ragu, 'cari'.
"""
import json
import re
import unicodedata
import zlib
from pathlib import Path

import numpy as np

import memory_tools as T

LABELS = ["urut", "hitung", "tanggal", "cari", "sapaan", "tolak"]
N_BUCKETS = 1 << 18
WEIGHTS_FILE = Path(__file__).with_name("router_weights.npz")

_NUMBER_WORDS = set(T._UNITS) | set(T._MULTS) | set(T._STRUCT) | {"se" + m for m in T._MULTS}
_MONTHS = "|".join(T.MONTHS).lower() + "|" + "jan|feb|mar|apr|jun|jul|agu|sep|okt|nov|des"
_DATE = re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}\b|\b\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}\b")
_MONTH_WORD = re.compile(rf"\b(?:{_MONTHS})\b")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_TOKEN = re.compile(r"<[a-z]+>|[a-z]+|[?=+*/^]")


def normalize(text):
    """Huruf kecil, tanpa aksen; tanggal menjadi <date>, bulan <month>, angka <n>, kata bilangan <nw>."""
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = _DATE.sub(" <date> ", t)
    t = _MONTH_WORD.sub(" <month> ", t)
    t = _NUM.sub(" <n> ", t)
    t = re.sub(r"\bx\b", " * ", t)                                   # 'x' sebagai tanda kali
    toks = _TOKEN.findall(t)
    return ["<nw>" if w in _NUMBER_WORDS else w for w in toks]


def _h(s):
    return zlib.crc32(s.encode("utf-8")) & (N_BUCKETS - 1)


def features(text):
    """Indeks fitur (di-hash) dan nilainya (biner, dinormalisasi L2). Sengaja TANPA fitur panjang kalimat (hindari jalan pintas)."""
    toks = normalize(text)
    feats = set()
    padded = ["<s>"] + toks + ["</s>"]
    for w in padded:
        feats.add("w:" + w)
    for a, b in zip(padded, padded[1:]):
        feats.add("b:" + a + " " + b)
    for w in toks:
        if w.startswith("<") or len(w) < 2:
            continue
        s = "#" + w + "#"
        for n in (3, 4, 5):
            for i in range(len(s) - n + 1):
                feats.add("c:" + s[i:i + n])
    # fitur hitungan JENIS ISI (bukan panjang kalimat): berapa tanggal, angka, dan kata bilangan; membantu pesan berstruktur seperti
    # 'beda <date> dengan <date> berapa hari' yang kata-katanya (dengan, berapa) lebih sering muncul pada pesan hitung
    for kind, tag in (("date", "<date>"), ("n", "<n>"), ("nw", "<nw>")):
        k = toks.count(tag)
        feats.add(f"k:{kind}:{min(k, 4) if kind != 'date' else min(k, 2)}")
    idx = np.fromiter((_h(f) for f in feats), dtype=np.int64)
    idx = np.unique(idx)
    return idx, np.full(len(idx), 1.0 / np.sqrt(max(1, len(idx))), dtype=np.float32)


class Router:
    def __init__(self, W, b, meta=None):
        self.W, self.b, self.meta = W.astype(np.float32), b.astype(np.float32), meta or {}

    @classmethod
    def load(cls, path=WEIGHTS_FILE):
        z = np.load(path, allow_pickle=False)
        return cls(z["W"], z["b"], json.loads(str(z["meta"])))

    def save(self, path=WEIGHTS_FILE):
        np.savez_compressed(path, W=self.W.astype(np.float16), b=self.b.astype(np.float16), meta=np.array(json.dumps(self.meta, ensure_ascii=False)))

    def logits(self, text):
        idx, val = features(text)
        return val @ self.W[idx] + self.b

    def proba(self, text):
        z = self.logits(text)
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def predict(self, text):
        p = self.proba(text)
        return LABELS[int(p.argmax())], p

    def decide(self, text, tau=0.5, tau_greeting=None, tau_refuse=None):
        """Kelas akhir dengan kebijakan hati-hati: kelas non-'cari' hanya bila probabilitasnya >= ambang kelas itu; selain itu 'cari'."""
        label, p = self.predict(text)
        if label == "cari":
            return "cari", p
        need = {"sapaan": tau_greeting, "tolak": tau_refuse}.get(label) or tau
        return (label, p) if p[LABELS.index(label)] >= need else ("cari", p)
