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


# ---------- dipakai aplikasi: classifier MENGUSULKAN, alat deterministik MEMVERIFIKASI ----------
import os
import json as _json

MODE_ENV = "MEMORI_ROUTER"            # off (bawaan, perilaku lama) | shadow (jalan paralel, hanya mencatat) | on
SHADOW_LOG_ENV = "MEMORI_ROUTER_LOG"  # berkas log lokal; bawaan hanya berisi kelas dan panjang pesan, TIDAK menyimpan teks pesan
SHADOW_TEXT_ENV = "MEMORI_ROUTER_LOG_TEXT"  # "1" = ikut simpan teks pesan di berkas lokal itu (untuk dilabeli pemilik); mati bila tidak diisi
_ROUTER = None
CONFIG_ENV = "MEMORI_CONFIG"          # jalur berkas konfigurasi; bila kosong dicari memori.env lalu .env di folder proyek, lalu di folder kerja
_CONFIG_NAMES = ("memori.env", ".env")
_CONFIG_CACHE = None


def _config_paths():
    explicit = os.environ.get(CONFIG_ENV)
    if explicit:
        return [Path(explicit)]
    roots = [Path(__file__).resolve().parent.parent, Path.cwd()]
    return [r / n for r in roots for n in _CONFIG_NAMES]


def parse_config(text):
    """Baris KEY=VALUE; '#' memulai komentar di awal baris; tanda kutip di sekeliling nilai dibuang; baris lain diabaikan."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'" + chr(34):
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].strip()
        if k:
            out[k] = v
    return out


def reload_config():
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def setting(name, default=None):
    """Urutan prioritas: variabel lingkungan sungguhan > berkas konfigurasi (memori.env atau .env) > bawaan."""
    global _CONFIG_CACHE
    if name in os.environ:
        return os.environ[name]
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = {}
        for path in _config_paths():
            try:
                if path.is_file():
                    _CONFIG_CACHE = parse_config(path.read_text(encoding="utf-8-sig"))
                    break
            except OSError:
                pass
    return _CONFIG_CACHE.get(name, default)


def _threshold(name, default):
    try:
        v = float(setting(name, default))
        return v if 0.0 < v <= 1.0 else default
    except (TypeError, ValueError):
        return default


def _router():
    global _ROUTER
    if _ROUTER is None:
        try:
            _ROUTER = Router.load()
        except Exception:
            _ROUTER = False           # bobot tidak ada atau rusak: aplikasi tetap jalan dengan aturan lama
    return _ROUTER or None


def get_mode(mode=None):
    m = (mode or setting(MODE_ENV, "off") or "off").strip().lower()
    return m if m in ("off", "shadow", "on") else "off"


def _decide(text):
    r = _router()
    if r is None:
        return None, None
    m = r.meta
    label, p = r.decide(text, m.get("tau", 0.5), m.get("tau_sapaan"), m.get("tau_tolak"))
    return label, p


def _rule_label(result):
    if result is None:
        return "cari"
    if result.get("hint"):
        return result["hint"]
    if result["kind"] == "task":
        return "tolak"
    t = result.get("tool") or ""
    return "urut" if "urut" in t or "angka" in t else "tanggal" if "tanggal" in t else "hitung"


def _shadow_log(rule_label, label, p, text):
    path = setting(SHADOW_LOG_ENV)
    if not path:
        return
    try:
        rec = {"aturan": rule_label, "classifier": label, "p_max": round(float(max(p)), 3), "kata": len(text.split())}
        if (setting(SHADOW_TEXT_ENV, "") or "").strip() == "1":
            rec["teks"] = text.strip()
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


HINT_MIN_P = 0.6        # arahan 'alat tak bisa membaca' hanya bila classifier cukup yakin
REFUSE_MIN_P = 0.6      # 'tolak' lebih berbahaya salah daripada kelas lain (mengalihkan pertanyaan catatan): ambang lebih tinggi
_MENTIONS_NOTES = re.compile(r"\b(catatan|dokumen|memori)\b", re.I)   # menyebut catatan: kemungkinan besar pertanyaan catatan, bukan permintaan tugas
_UNREADABLE = {
    "urut": "Sepertinya Anda meminta mengurutkan angka, tetapi saya tidak bisa membaca angkanya. Tulis angkanya langsung, mis. 'urutkan 5, 3, 9, 1'.",
    "hitung": "Sepertinya Anda meminta menghitung, tetapi saya tidak bisa membaca soalnya. Tulis hitungannya langsung, mis. 'hitung 12 x 5' atau '250000 - 87000'.",
    "tanggal": "Sepertinya Anda menanyakan hari atau selisih tanggal, tetapi saya butuh tanggal lengkap dengan tahun, mis. 'hari apa 17 Agustus 2026' atau 'selisih 15 Juni 2026 dan 2 Juli 2026'.",
}


def assisted_route(q, lex=None, notes=None, mode=None):
    """Pengganti memory_tools.route. Mode 'on': bila aturan ketat tidak menjawab, classifier boleh mengusulkan urut/hitung/tanggal
    (alat mencoba bacaan longgar dan tetap harus berhasil) atau tolak (permintaan tugas). Hasil alat yang sudah ada tidak pernah diubah."""
    rule = T.route(q, lex, notes)
    mode = get_mode(mode)
    if mode == "on" and rule is not None and rule.get("kind") == "task":
        # aturan lama menolak semua yang diawali kata kerja tugas ('hitung ...', 'jumlahkan ...'), termasuk permintaan alat yang tidak terbaca
        label, p = _decide(q)
        if label in ("urut", "hitung", "tanggal"):
            r = T.route(q, lex, notes, hint=label)
            if r is not None and r.get("kind") != "task":
                return r
            if float(p[LABELS.index(label)]) >= _threshold("MEMORI_ROUTER_HINT_MIN_P", HINT_MIN_P):
                return {"kind": "task", "tool": None, "answer": _UNREADABLE[label], "detail": "", "hint": label}
        return rule
    if mode == "off" or rule is not None:
        if mode == "shadow":
            label, p = _decide(q)
            if label is not None:
                _shadow_log(_rule_label(rule), label, p, q)
        return rule
    label, p = _decide(q)
    if label is None:
        return rule
    if mode == "shadow":
        _shadow_log("cari", label, p, q)
        return rule
    if label in ("urut", "hitung", "tanggal"):
        r = T.route(q, lex, notes, hint=label)
        if r is None and float(p[LABELS.index(label)]) >= _threshold("MEMORI_ROUTER_HINT_MIN_P", HINT_MIN_P):   # yakin ini permintaan alat tetapi alat tak bisa membacanya: beri arahan, bukan 'belum tahu' dari catatan
            return {"kind": "task", "tool": None, "answer": _UNREADABLE[label], "detail": "", "hint": label}
        return r
    if label == "tolak" and len(q.split()) <= 12 and float(p[LABELS.index("tolak")]) >= _threshold("MEMORI_ROUTER_REFUSE_MIN_P", REFUSE_MIN_P) and not _MENTIONS_NOTES.search(q):
        return {"kind": "task", "tool": None, "answer": T.TASK_MESSAGE, "detail": ""}
    return None


_GREET_STEMS = ("halo", "hai", "hi", "hei", "hello", "hallo", "selamat", "terima", "kasih", "makasih", "makasi", "mksh", "thanks",
                "thank", "thx", "salam", "assalamualaikum", "wassalamualaikum", "waalaikumsalam", "permisi", "dadah", "bye", "sampai", "jumpa", "pamit", "kenal",
                "datang", "kabar", "oke", "sip", "sob", "bro", "kak", "tim")
_TIME_WORDS = {"pagi", "siang", "sore", "malam"}
_REPEAT = re.compile(r"(.)\1+")


def has_greeting_cue(q):
    """Penjaga: classifier boleh mengusulkan 'sapaan' hanya bila ada kata yang mirip kosakata sapaan/terima kasih/pamit (toleran salah ketik dan huruf berulang).
    Mencegah pesan pendek tak berkaitan (mis. 'pandora', 'bangunkan saya pukul tujuh') dibalas dengan sapaan."""
    import difflib
    for tok in re.findall(r"[a-z]+", q.lower()):
        for form in {tok, _REPEAT.sub(r"\1", tok)}:
            if form in _GREET_STEMS:
                return True
            close = difflib.get_close_matches(form, _GREET_STEMS, n=1, cutoff=0.79) if len(form) >= 4 else []
            if close and close[0][0] == form[0]:         # huruf pertama harus sama: menolak 'alarm' ~ 'salam'
                return True
    return len(q.split()) <= 3 and any(w in _TIME_WORDS for w in re.findall(r"[a-z]+", q.lower()))   # 'sore pak'; tidak untuk kalimat panjang ('bangunkan saya pukul tujuh pagi')


def assisted_greeting(q, mode=None):
    """True bila classifier (mode 'on') yakin ini sapaan pendek yang lolos dari pola bawaan (mis. salah ketik 'selamt pagi') DAN ada petunjuk kata sapaan."""
    if get_mode(mode) != "on" or len(q.split()) > 8 or any(ch.isdigit() for ch in q) or not has_greeting_cue(q):
        return False
    label, _ = _decide(q)
    return label == "sapaan"
