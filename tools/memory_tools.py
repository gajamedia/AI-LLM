"""
Alat deterministik untuk Memori: pekerjaan yang dikerjakan KODE, bukan model.

Model 100M hanya bisa menyalin potongan catatan; ia tidak bisa berhitung atau mengurutkan. Alat ini menangani
sejumlah pola pasti dan hasilnya diberi label "dihitung oleh alat (bukan dari catatan)":
  - pengurut angka   : "urutkan 5, 4, 3, 9", "urutkan deret angka ini (5-4-3-9) dari besar ke kecil"
  - kalkulator       : "berapa 3 + 4", "hitung 12 x 5", "100 dibagi 4", "2 pangkat 10"
  - tanggal          : "hari apa 17 Agustus 2026", "berapa hari antara 1 Januari 2026 dan 1 Maret 2026"
route(teks) -> dict atau None:
  {"kind": "tool", "tool": nama, "answer": hasil, "detail": penjelasan}
  {"kind": "task", "tool": None, "answer": pesan, "detail": ""}   (permintaan tugas yang tidak didukung)
Dirancang agar TIDAK membajak pertanyaan biasa tentang catatan (mis. "lahir tanggal 12-03-1990?" bukan pengurangan).
"""
import ast
import datetime as dt
import operator
import re
from pathlib import Path

TOOL_LIST = ("mengurutkan angka, menghitung (+ - x / ^), menulis angka dengan huruf (mis. 10), hari/selisih tanggal, "
             "dan mengolah angka di satu catatan (mis. 'jumlahkan angka di catatan 3')")
TASK_MESSAGE = ("Ini permintaan tugas, bukan pertanyaan tentang catatan Anda. Aplikasi ini hanya membaca catatan yang "
                "Anda ajarkan, jadi tidak bisa menulis, menerjemahkan, atau merangkum. Alat yang tersedia: " + TOOL_LIST + ".")

MONTHS = {"januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
          "september": 9, "oktober": 10, "november": 11, "desember": 12}
DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
TASK_VERBS = (r"(urutkan|mengurutkan|sortir|hitung|hitunglah|jumlahkan|kalikan|terjemahkan|tuliskan|tulis|buatkan|buat(?!\s+apa\b)|bikin|"
              r"ringkaskan|rangkum|ceritakan|karang|mengarang|gubah|ubah|susun|translate)")


# ---------- angka ----------
def _format_number(x):
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        x = int(x)
    if isinstance(x, int):
        return str(x)
    s = f"{x:.10g}"
    return s.replace(".", ",")


def _normalize_numbers(text, hyphen_is_separator):
    """Samakan penulisan angka: ribuan 1.000 -> 1000, desimal 3,5 -> 3.5, daftar 5,4,3 -> pemisah."""
    t = text
    t = re.sub(r"\d+(?:,\d+){2,}", lambda m: m.group(0).replace(",", " "), t)       # 5,4,3,9 = daftar
    t = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b(?!\d)", lambda m: m.group(0).replace(".", ""), t)  # ribuan
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)                                           # desimal koma
    if hyphen_is_separator:
        t = re.sub(r"(?<=\d)-(?=\d)", " ", t)
    return t


def parse_numbers(text, hyphen_is_separator=True):
    t = _normalize_numbers(text, hyphen_is_separator)
    out = []
    for tok in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", t):
        out.append(float(tok) if "." in tok else int(tok))
    return out


BASE_LEXICON_FILE = Path(__file__).with_name("kamus_bawaan.txt")
_BASE_LINE = re.compile(r'^(kata|pengali|operator)\s+"([a-z]{2,20})"\s+berarti\s+(\S+)$')
_BASE_OPS = {"+": "+", "-": "-", "*": "*", "/": "/", "^": "^"}


def load_base_lexicon(path=BASE_LEXICON_FILE):
    """Baca kamus bawaan dari berkas data (bukan kode). Kembalikan (unit, pengali, operator)."""
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as e:
        raise RuntimeError(f"Kamus bawaan tidak ditemukan atau tidak terbaca ({path}): {e}. Berkas kamus_bawaan.txt harus ada di folder tools.") from e
    units, mults, ops = {}, {}, {}
    for n, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _BASE_LINE.match(line)
        ok = bool(m)
        if ok:
            kind, word, val = m.groups()
            if kind == "kata" and re.fullmatch(r"\d", val):
                units[word] = int(val)
            elif kind == "pengali" and re.fullmatch(r"10{3,12}", val):
                mults[word] = int(val)
            elif kind == "operator" and val in _BASE_OPS:
                ops[word] = val
            else:
                ok = False
        if not ok:
            raise RuntimeError(f"Kamus bawaan {path}, baris {n} tidak valid: {raw!r}")
    if not units or not mults:
        raise RuntimeError(f"Kamus bawaan {path} tidak memuat kata bilangan atau pengali.")
    return units, mults, ops


_UNITS, _MULTS, _BASE_OPERATORS = load_base_lexicon()
_STRUCT = ["sepuluh", "sebelas", "belas", "puluh", "seratus", "ratus"]
_RUN_CACHE = {}


def _run_regex(units, mults):
    key = (tuple(sorted(units)), tuple(sorted(mults)))
    rx = _RUN_CACHE.get(key)
    if rx is None:
        words = sorted(set(units) | set(mults) | {"se" + m for m in mults} | set(_STRUCT), key=len, reverse=True)
        alt = "|".join(re.escape(w) for w in words)
        rx = re.compile(r"\b(?:" + alt + r")\b(?:\s+(?:" + alt + r")\b)*", re.I)
        if len(_RUN_CACHE) > 32:
            _RUN_CACHE.clear()
        _RUN_CACHE[key] = rx
    return rx


def _parse_number_run(tokens, units=None, mults=None):
    """Ubah deret kata bilangan (mis. 'dua puluh lima', 'dua juta tiga ratus ribu', 'tiga satu dua') menjadi daftar bilangan bulat."""
    units = _UNITS if units is None else units
    mults = _MULTS if mults is None else mults
    nums, total, sub, cur, last, lm = [], 0, 0, None, None, None  # lm: pengali terakhir (harus menurun)

    def flush():
        nonlocal total, sub, cur, last, lm
        if last is not None:
            nums.append(total + sub + (cur or 0))
        total, sub, cur, last, lm = 0, 0, None, None, None

    for t in tokens:
        if t in units:
            if last in ("unit", "teen"):
                flush()
            cur, last = units[t], "unit"
        elif t in ("sepuluh", "sebelas"):
            if last in ("unit", "teen", "tens"):
                flush()
            sub += 10 if t == "sepuluh" else 11
            last = "teen"
        elif t == "belas" and last == "unit":
            sub, cur, last = sub + cur + 10, None, "teen"
        elif t == "puluh" and last == "unit":
            sub, cur, last = sub + cur * 10, None, "tens"
        elif t == "seratus":
            if last in ("unit", "teen", "tens"):
                flush()
            sub, last = sub + 100, "hundreds"
        elif t == "ratus" and last == "unit":
            sub, cur, last = sub + cur * 100, None, "hundreds"
        elif t.startswith("se") and t[2:] in mults:      # seribu, sejuta, ...
            v = mults[t[2:]]
            if last is not None and not (lm is not None and v < lm and sub + (cur or 0) == 0):
                flush()
            total, lm, last = total + v, v, "mult"
        elif t in mults and last in ("unit", "teen", "tens", "hundreds"):
            v, pending = mults[t], sub + (cur or 0)
            if lm is not None and v >= lm:               # urutan naik (mis. 'dua ribu tiga juta'): bilangan baru
                nums.append(total)
                total = 0
            total, sub, cur, lm, last = total + pending * v, 0, None, v, "mult"
    flush()
    return nums


def words_to_digits(text, lex=None):
    """Ganti kata bilangan Indonesia dengan angka: 'dua puluh lima ditambah satu' -> '25 ditambah 1'.
    lex: kamus dari catatan pengguna (lihat parse_lexicon); menambah kata bilangan dan pengali."""
    units, mults = _UNITS, _MULTS
    if lex and (lex.get("units") or lex.get("multipliers")):
        units, mults = {**_UNITS, **lex.get("units", {})}, {**_MULTS, **lex.get("multipliers", {})}

    def repl(m):
        nums = _parse_number_run(m.group(0).lower().split(), units, mults)
        return " ".join(str(n) for n in nums) if nums else m.group(0)
    return _run_regex(units, mults).sub(repl, text)


# ---------- kamus dari catatan pengguna ----------
QUESTION_WORDS = {"apa", "apakah", "siapa", "siapakah", "kapan", "kapankah", "dimana", "dimanakah", "mana", "manakah",
                  "bagaimana", "bagaimanakah", "berapa", "berapakah", "mengapa", "kenapa", "bilamana", "kemana", "darimana"}
MAX_LEXICON = 200
_LEX_NOTE = re.compile(r"^\s*(kata|pengali|operator)\s+[\"“'‘](.+?)[\"”'’]\s+(?:berarti|artinya|sama dengan|adalah)\s+(\S+?)\s*[.!]?\s*$", re.I)
_LEX_HEAD = re.compile(r"^\s*(kata|pengali|operator)\s+[\"“'‘]", re.I)
_OP_SYMBOLS = {"+": "+", "-": "-", "*": "*", "x": "*", "×": "*", "/": "/", "÷": "/", "^": "^",
               "tambah": "+", "plus": "+", "kurang": "-", "minus": "-", "kali": "*", "bagi": "/", "pangkat": "^"}
_RESERVED = (QUESTION_WORDS | set(_UNITS) | set(_MULTS) | set(_BASE_OPERATORS) | set(_STRUCT) | set(MONTHS) | {d.lower() for d in DAYS} | {
    "halo", "hai", "hei", "hello", "selamat", "terima", "kasih", "makasih", "thanks", "kabar", "pagi", "siang", "sore", "malam",
    "ditambah", "tambah", "plus", "dikurangi", "minus", "dikali", "dikalikan", "kali", "dibagi", "bagi", "pangkat", "kurang",
    "urutkan", "mengurutkan", "sortir", "sort", "hitung", "hitunglah", "jumlahkan", "kalikan", "terjemahkan", "tuliskan",
    "buatkan", "ringkaskan", "rangkum", "ceritakan", "translate", "hasil", "hasilnya", "tolong", "coba", "selisih", "jarak",
    "yang", "dan", "di", "ke", "dari", "atau", "ini", "itu", "untuk", "dengan", "pada", "adalah", "ya", "tidak", "bukan",
    "ingat", "lupakan", "daftar", "tanya", "dokumen", "impor", "hapus", "bantuan", "kata", "pengali", "operator", "berarti", "artinya"})


def _parse_entry(text):
    """(jenis, kata, nilai) dari satu catatan format kamus; atau str berisi alasan penolakan; atau None bila bukan format kamus."""
    if not _LEX_HEAD.match(text):
        return None
    m = _LEX_NOTE.match(text)
    if not m:
        return 'Format kamus: kata "siji" berarti 1  |  pengali "ewu" berarti 1000  |  operator "campur" berarti +'
    kind, word, val = m.group(1).lower(), m.group(2).strip().lower(), m.group(3).strip().lower()
    if not re.fullmatch(r"[a-z]{2,20}", word):
        return f'Kata "{word}" tidak valid: harus 2-20 huruf a-z tanpa spasi, angka, atau tanda baca.'
    if word in _RESERVED:
        return f'Kata "{word}" sudah dipakai program (kata tanya, sapaan, perintah, bulan/hari, atau bilangan/operator bawaan); pilih kata lain.'
    if kind == "kata":
        if not re.fullmatch(r"\d", val):
            return f'Nilai kata harus satu angka 0-9, bukan "{val}". Bilangan besar dibentuk dari kata-kata ini ("siji" "loro" ...).'
        return kind, word, int(val)
    if kind == "pengali":
        digits = val.replace(".", "")
        if not re.fullmatch(r"10{3,12}", digits):
            return 'Nilai pengali harus kelipatan sepuluh dari 1000 sampai 1000000000000 (mis. 1000, 1000000).'
        return kind, word, int(digits)
    sym = _OP_SYMBOLS.get(val)
    if sym is None:
        return 'Nilai operator harus salah satu dari: + - * / ^ (atau tambah, kurang, kali, bagi, pangkat).'
    return kind, word, sym


_PAIR = re.compile(r"(?<![\w.])(\d)\s*=\s*([A-Za-z]{2,20})\b")


def _parse_table(text):
    """Daftar bebas 'angka = kata' (mis. 'Daftar Bilangan 1 Sampai 10 \u2022 1 = satu \u2022 2 = dua ...').
    Kembalikan (entri, dilewati) atau None bila bukan daftar (butuh minimal 2 pasangan)."""
    if _LEX_HEAD.match(text):
        return None
    pairs = _PAIR.findall(text)
    if len(pairs) < 2:
        return None
    entries, skipped = [], []
    for d, w in pairs:
        w, d = w.lower(), int(d)
        if w in _UNITS:
            if _UNITS[w] != d:
                skipped.append(f'"{w}" (bawaan bernilai {_UNITS[w]}, bukan {d})')
        elif w in _RESERVED:
            skipped.append(f'"{w}" (kata sudah dipakai program)')
        else:
            entries.append(("kata", w, d))
    return entries, skipped


def parse_lexicon(notes):
    """notes: daftar (id, teks). Kembalikan (kamus, masalah). Catatan yang lebih baru menimpa yang lama."""
    lex = {"units": {}, "multipliers": {}, "operators": {}}
    keys = {"kata": "units", "pengali": "multipliers", "operator": "operators"}
    issues, count = [], 0
    for nid, text in sorted(notes, key=lambda n: n[0]):
        e = _parse_entry(text)
        if e is None:
            tbl = _parse_table(text)
            if tbl is None:
                continue
            entries = tbl[0]
        elif isinstance(e, str):
            issues.append((nid, e))
            continue
        else:
            entries = [e]
        for kind, word, val in entries:
            if count >= MAX_LEXICON:
                issues.append((nid, f"Batas {MAX_LEXICON} entri kamus tercapai; sisanya diabaikan."))
                break
            count += 1
            for k, d in lex.items():
                d.pop(word, None)            # satu kata hanya punya satu arti: yang terbaru menang
            lex[keys[kind]][word] = val
    return lex, issues


def describe_lexicon_note(text):
    """None bila catatan bukan format kamus; selain itu {'ok': bool, 'message': str} untuk umpan balik saat mengajar."""
    e = _parse_entry(text)
    if e is None:
        tbl = _parse_table(text)
        if tbl is None:
            return None
        new, skipped = tbl
        msg = ("Dikenali sebagai daftar kamus: " + (f"{len(new)} kata baru (" + ", ".join(f'{w}={v}' for _, w, v in new[:6]) + (", ..." if len(new) > 6 else "") + "). Alat urut/hitung kini memahaminya."
               if new else "semua kata di dalamnya sudah dikenal program."))
        if skipped:
            msg += " Dilewati: " + "; ".join(skipped[:4]) + "."
        return {"ok": True, "message": msg}
    if isinstance(e, str):
        return {"ok": False, "message": e + " Catatan tetap disimpan tetapi tidak dipakai sebagai kamus."}
    kind, word, val = e
    what = {"kata": f'kata bilangan "{word}" = {val}', "pengali": f'pengali "{word}" = {val}',
            "operator": f'operator "{word}" = {val}'}[kind]
    return {"ok": True, "message": f"Dikenali sebagai kamus: {what}. Alat urut/hitung kini memahaminya."}


SORT_VERB = re.compile(r"\b(urutkan|mengurutkan|diurutkan|urutan|berurutan|sortir|sort|susunlah|susun|menyusun|disusun|rapikan|merapikan)\b", re.I)
_SORT_RANGE = re.compile(r"(?:ter)?kecil\s*(?:ke|->|-|sampai|hingga)\s*(?:ter)?besar|(?:ter)?besar\s*(?:ke|->|-|sampai|hingga)\s*(?:ter)?kecil", re.I)
_ASC_CUE = re.compile(r"(?:ter)?kecil\s*(?:ke|->|-|sampai|hingga)\s*(?:ter)?besar|menaik|\bnaik\b|ascending|dari\s+(?:yang\s+)?(?:paling\s+)?(?:ter)?kecil", re.I)
_DESC_CUE = re.compile(r"terbesar|(?:ter)?besar\s*(?:ke|->|-|sampai|hingga)\s*(?:ter)?kecil|menurun|\bturun\b|descending|dari\s+yang\s+besar", re.I)


def wants_desc(q):
    """Urutan menurun? Isyarat menaik ('dari kecil', 'terkecil ke terbesar') didahulukan agar 'terbesar' di ujung rentang tidak membalik urutan."""
    return False if _ASC_CUE.search(q) else bool(_DESC_CUE.search(q))


def sort_tool(q, lex=None):
    if not (SORT_VERB.search(q) or _SORT_RANGE.search(q)):
        return None
    nums = parse_numbers(words_to_digits(q, lex))
    if len(nums) < 2:
        return None
    desc = wants_desc(q)
    ordered = sorted(nums, reverse=desc)
    sep = "; " if any(isinstance(x, float) and not x.is_integer() for x in nums) else ", "  # koma desimal vs pemisah
    return {"kind": "tool", "tool": "pengurut angka", "answer": sep.join(_format_number(x) for x in ordered),
            "detail": f"Angka dibaca: {sep.join(_format_number(x) for x in nums)} lalu diurutkan {'menurun (besar ke kecil)' if desc else 'menaik (kecil ke besar)'}."}


# ---------- kalkulator ----------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}


class TooBig(ValueError):
    """Perhitungan yang terlalu besar untuk alat ini (dicegah agar tidak membekukan aplikasi)."""


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _safe_eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(b) > 64 or abs(a) > 1e6:
                raise TooBig("pangkat terlalu besar")
        return _OPS[type(node.op)](a, b)
    raise ValueError("ekspresi tidak didukung")


_FILLER = {"berapa", "berapakah", "hitung", "hitunglah", "hasil", "hasilnya", "dari", "adalah", "ini", "itu", "tolong",
           "yang", "ya", "saja", "dong", "sama", "dengan", "nya", "jumlah", "jumlahkan", "kalikan", "bagi", "kurang", "kurangi"}
_TRIGGER = re.compile(r"\b(hitung|hitunglah|berapa|berapakah|hasil|hasilnya|sisanya?|jumlahkan|kalikan)\b", re.I)


_N = r"\d+(?:[.,]\d+)?"


def _prefix_verbs(t):
    """Kata kerja di depan angka -> bentuk tengah: 'tambahkan 25 dengan 17' -> '25 + 17'; 'total dari 7 dan 15 dan 20' -> '7 + 15 + 20'."""
    t = re.sub(rf"\b(?:tambahkan|jumlahkan|totalkan)\s+({_N})\s+(?:dengan|dan|ke)\s+({_N})\b", r"\1 + \2", t)
    t = re.sub(rf"\bkurangi\s+({_N})\s+(?:dengan|oleh)\s+({_N})\b", r"\1 - \2", t)
    t = re.sub(rf"\bkalikan\s+({_N})\s+(?:dengan|dan)\s+({_N})\b", r"\1 * \2", t)

    def _sum(m):
        nums = re.findall(_N, m.group(1))
        return " + ".join(nums) + " " if len(nums) >= 2 else m.group(0)
    return re.sub(rf"\b(?:jumlahkan|jumlah|totalnya|total)(?:\s+dari)?\s+((?:{_N}\s*(?:,|dan)?\s*)+)", _sum, t)


def calc_tool(q, lex=None):
    t = _prefix_verbs(words_to_digits(q.lower(), lex)).strip().rstrip("?= ")
    if re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", t) or re.search(r"\b\d{3,}-\d{3,}", t):
        return None  # tanggal atau nomor telepon, bukan pengurangan
    expr_text = t
    for word, sym in {**_BASE_OPERATORS, **((lex or {}).get("operators") or {})}.items():  # bawaan (berkas data) + ajaran pengguna
        expr_text = re.sub(r"\b" + re.escape(word) + r"\b", " ** " if sym == "^" else f" {sym} ", expr_text)
    expr_text = re.sub(r"(?<=[\d)])\s*[x×]\s*(?=[\d(])", " * ", expr_text)
    expr_text = expr_text.replace("÷", " / ").replace("^", " ** ").replace(":", " : ")
    expr_text = _normalize_numbers(expr_text, hyphen_is_separator=False)
    cands = re.findall(r"[\d\.\s\+\-\*/\(\)%]+", expr_text)
    cands = [c.strip() for c in cands if re.search(r"\d\s*[\+\-\*/%]+\s*[\(\-]?\s*\d", c) or re.search(r"\d\s*\)?\s*[\+\*/%]", c)]
    if not cands:
        return None
    expr = max(cands, key=len)
    leftover = [w for w in re.findall(r"[a-zA-Z]+", re.sub(re.escape(expr), " ", expr_text)) if w not in _FILLER]
    whole_expression = not re.search(r"[a-zA-Z]", expr_text)
    if not (_TRIGGER.search(q) or whole_expression) or len(leftover) > 2:
        return None
    try:
        value = _safe_eval(ast.parse(expr.replace("%", " % "), mode="eval"))
    except ZeroDivisionError:
        return {"kind": "tool", "tool": "kalkulator", "answer": "Tidak bisa dibagi nol.", "detail": f"Ekspresi: {expr}"}
    except TooBig:
        return {"kind": "tool", "tool": "kalkulator", "answer": "Terlalu besar untuk alat ini (pangkat maksimal 64, bilangan dasar maksimal 1.000.000).",
                "detail": f"Ekspresi: {expr}"}
    except (ValueError, SyntaxError, OverflowError, RecursionError):
        return None
    shown = re.sub(r"\s+", " ", expr).replace("**", "^")
    return {"kind": "tool", "tool": "kalkulator", "answer": _format_number(value), "detail": f"Ekspresi dibaca: {shown}"}


# ---------- terbilang (angka -> nama bilangan, nama dari kamus) ----------
_TERBILANG = re.compile(
    r"^\s*(?:terbilang\s+|tulis(?:kan)?\s+|eja(?:kan)?\s+)?(\d{1,3}(?:\.\d{3})+|\d+)\s*"
    r"(?:=\s*\??|dibaca\s+apa\??|dalam\s+(?:kata|huruf)\??|\?)?\s*$", re.I)


def _names(lex):
    """Nama untuk tiap nilai: yang lebih dulu didefinisikan menang (bawaan lebih dulu daripada catatan pengguna)."""
    un, mn = {}, {}
    for w, v in {**_UNITS, **((lex or {}).get("units") or {})}.items():
        un.setdefault(v, w)
    for w, v in {**_MULTS, **((lex or {}).get("multipliers") or {})}.items():
        mn.setdefault(v, w)
    return un, mn


def spell_number(n, lex=None):
    """Nama bilangan bulat 0 <= n < 1000 * pengali terbesar, mis. 2300 -> 'dua ribu tiga ratus'."""
    un, mn = _names(lex)

    def below_1000(k):
        out = []
        h, r = divmod(k, 100)
        if h == 1:
            out.append("seratus")
        elif h:
            out.append(f"{un[h]} ratus")
        if 10 <= r < 20:
            out.append("sepuluh" if r == 10 else "sebelas" if r == 11 else f"{un[r - 10]} belas")
        else:
            t, u = divmod(r, 10)
            if t:
                out.append(f"{un[t]} puluh")
            if u:
                out.append(un[u])
        return " ".join(out)

    def spell(k):
        if k == 0:
            return un[0]
        parts = []
        for v in sorted(mn, reverse=True):
            q, k = divmod(k, v)
            if q:
                parts.append(f"se{mn[v]}" if (v == 1000 and q == 1) else f"{below_1000(q)} {mn[v]}")
        if k:
            parts.append(below_1000(k))
        return " ".join(parts)

    return spell(n)


def terbilang_tool(q, lex=None):
    m = _TERBILANG.match(q)
    if not m:
        return None
    n = int(m.group(1).replace(".", ""))
    un, mn = _names(lex)
    if n >= 1000 * max(mn):
        return None
    try:
        words = spell_number(n, lex)
    except KeyError:
        return None  # kamus tidak punya nama untuk salah satu angka yang dibutuhkan
    return {"kind": "tool", "tool": "terbilang", "answer": words,
            "detail": f"Angka {n} dibaca: {words} (nama bilangan dari kamus)."}


# ---------- tanggal ----------
def _dates(q):
    out = []
    for m in re.finditer(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b", q, re.I):
        try:
            out.append(dt.date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1))))
        except ValueError:
            return []
    return out


def date_tool(q):
    ds = _dates(q)
    if len(ds) == 1 and re.search(r"\bhari apa(?:kah)?\b", q, re.I):
        d = ds[0]
        return {"kind": "tool", "tool": "tanggal", "answer": DAYS[d.weekday()],
                "detail": f"{d.day} {[k for k, v in MONTHS.items() if v == d.month][0].capitalize()} {d.year} jatuh pada hari {DAYS[d.weekday()]}."}
    if len(ds) == 2 and re.search(r"\b(berapa hari|berapa lama|selisih|jarak|rentang|beda)\b", q, re.I):
        n = abs((ds[1] - ds[0]).days)
        return {"kind": "tool", "tool": "tanggal", "answer": f"{n} hari", "detail": f"Selisih antara {ds[0].isoformat()} dan {ds[1].isoformat()}."}
    return None


# ---------- sapaan dan saran mengajar (dipakai CLI dan web) ----------
INTRO = ("AI ini hanya tahu apa yang Anda ajarkan; ia tidak menjawab dari pengetahuan umum.\n"
         "Mulai dengan:  ingat: Bu Rina tinggal di Jalan Mawar nomor 5.\n"
         "Lalu tanya:    Di mana Bu Rina tinggal?")
_GREET_TAIL = r"(?:\s+(?:apa\s+kabar|semua|semuanya|teman|sobat|kak|kakak|pak|bu|mas|mbak|bro|sis|ya|banyak|lagi))*"
GREETING = re.compile(
    r"^(?:(?:halo|hai|hi|hei|hello|selamat\s+(?:pagi|siang|sore|malam|datang)|apa\s+kabar|terima\s+kasih|makasih|salam"
    r"|thanks?(?:\s+a\s+lot)?|thank\s+you)" + _GREET_TAIL
    + r"|assalamu\S*(?:\s+(?:warahmatullahi|wabarakatuh|wr|wb|wa)\S*)*)\W*$", re.I)
TABLE_HINT = "Tabel atau rumus tidak bisa dipelajari dari catatan. Untuk menghitung, ketik langsung, mis. 'hitung 12 x 5'."


def greeting_reply(n_notes, n_docs=0):
    base = "Halo! Saya memori pribadi Anda: saya hanya tahu apa yang Anda ajarkan, bukan pengetahuan umum.\n"
    if n_notes == 0 and n_docs == 0:
        return base + INTRO
    docs = f" dan {n_docs} dokumen" if n_docs else ""
    return base + f"Saat ini ada {n_notes} catatan{docs}. Tanyakan dengan kalimat berakhiran '?', atau tambah catatan dengan 'ingat: ...'."


_ODD_SYMBOLS = re.compile(r"[²³¹⁰-⁹¼-¾≈≤≥→]")
LONG_NOTE_CHARS = 400


def vocab_warnings(text):
    """Peringatan khusus model kecil: simbol di luar kosakata dan catatan yang mungkin melebihi konteks 256 token."""
    out = []
    if _ODD_SYMBOLS.search(text):
        out.append("Simbol seperti ² ³ ½ tidak dikenal model dan merusak jawaban. Tulis 'km2' atau 'km persegi', bukan 'km²'.")
    if len(text) > LONG_NOTE_CHARS:
        out.append(f"Catatan {len(text)} karakter mungkin melebihi konteks model (256 token); sebagian akan dipotong saat dibaca, "
                   "dan angka panjang sering salah disalin. Pecah menjadi satu fakta per catatan.")
    return out


MAX_MULTI_CHARS = 6000
MAX_MULTI_LINES = 60
_BULLET = re.compile(r"^(?:[-*\u2022\u2013]|\d{1,3}[.)])\s+")


def split_note_lines(text):
    """Pisahkan teks beberapa baris menjadi catatan terpisah (satu per baris). Penanda daftar (-, *, bullet, 1.) dibuang.
    Baris berisi satu kata (mis. judul 'Catatan:') dilewati. Kembalikan (daftar_baris, dilewati).
    Teks yang berupa daftar 'angka = kata' (kamus) tidak dipecah: itu satu catatan."""
    if _parse_table(text) is not None:
        return [re.sub(r"\s+", " ", text).strip()], []
    lines, skipped = [], []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _BULLET.sub("", raw.strip()).strip()
        if not line:
            continue
        (lines if len(line.split()) >= 2 else skipped).append(line)
    return lines, skipped


def looks_like_table(text):
    """Catatan berisi tabel/rumus hitung (mis. '1+1=2 1+2=3 ...'), yang tidak bisa dipelajari model."""
    return len(re.findall(r"\d\s*[+\-x×*/]\s*\d\s*=\s*\d", text)) >= 2 or text.count("=") >= 3


# ---------- pertanyaan lanjutan (riwayat percakapan) ----------
MAX_HISTORY = 5
_SKIP_FIRST = QUESTION_WORDS | {"di", "dan", "kalau", "kalo", "lalu", "terus", "kemudian", "bagaimana", "dengan", "tolong",
                                "coba", "ke", "dari", "untuk"}
_CONNECT = re.compile(r"^(?:dan|kalau|kalo|bagaimana\s+dengan|bagaimana\s+kalau|lalu|terus|kemudian)\b[\s,]*(.+?)\s*\??\s*$", re.I)
_PRONOUN = re.compile(r"\b(dia|ia|beliau|mereka)\b", re.I)
_NYA_WORD = re.compile(r"\b([a-z]{4,})nya\b")


def entities(q):
    """Frasa nama di pertanyaan: kata-kata berurutan berhuruf kapital (selain kata tanya di awal) atau berangka."""
    out, cur = [], []
    for i, w in enumerate(re.findall(r"[\w-]+", q)):
        if (w[:1].isupper() and not (i == 0 and w.lower() in _SKIP_FIRST)) or any(c.isdigit() for c in w):
            cur.append(w)
        elif cur:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def _as_question(text):
    text = text.strip().rstrip("?").strip()
    return text[:1].upper() + text[1:] + "?"


def _fill_reference(q, entity):
    """Ganti kata ganti (dia/ia/beliau/mereka) atau kata berakhiran -nya dengan nama; None bila tidak ada rujukan."""
    if entities(q):
        return None
    if _PRONOUN.search(q):
        return _PRONOUN.sub(entity, q, count=1)
    if _NYA_WORD.search(q):
        return _NYA_WORD.sub(lambda m: f"{m.group(1)} {entity}", q, count=1)
    return None


def rewrite_followup(question, history):
    """Ubah pertanyaan lanjutan menjadi pertanyaan lengkap memakai pertanyaan sebelumnya (history: daftar pertanyaan lengkap,
    terlama dulu). Hanya menulis ulang bila tandanya jelas; selain itu pertanyaan dikembalikan apa adanya.
    Kembalikan (pertanyaan, diubah: bool)."""
    q = question.strip()
    prev = next((h for h in reversed(history or []) if entities(h)), None)
    if not q or prev is None:
        return question, False
    old = entities(prev)[0]
    out = None
    m = _CONNECT.match(q)
    if m:
        rest = m.group(1)
        while _CONNECT.match(rest):                                 # "lalu kalau ..." : buang semua awalan penyambung
            rest = _CONNECT.match(rest).group(1)
        ne = entities(rest)
        first = (re.findall(r"[\w-]+", rest) or [""])[0].lower()
        if ne and len(rest.split()) <= len(ne[0].split()) + 1 and first not in QUESTION_WORDS:
            out = prev.replace(old, ne[0], 1)                       # "kalau Pak Joko?" -> pertanyaan lama untuk Pak Joko
        elif ne:
            out = _as_question(rest)                                # sudah lengkap, hanya awalan penyambung dibuang
        elif first in QUESTION_WORDS or first == "di":
            out = _fill_reference(_as_question(rest), old)
        elif len(rest.split()) <= 3:
            attr = _NYA_WORD.sub(lambda mm: mm.group(1), rest)      # "istrinya" -> "istri"
            out = _as_question(f"Apa {attr} {old}")
    else:
        out = _fill_reference(q, old)
    if not out or out.strip() == q:
        return question, False
    return out, True


# ---------- angka dari catatan tertentu ----------
_NOTE_REF = re.compile(r"\bcatatan\s+(?:#|nomor\s+|no\.?\s*)?(\d+)\b|\bcatatan\s+(terakhir|terbaru|paling\s+baru)\b", re.I)
_SORT_VERB = SORT_VERB
_NUM_WORD = re.compile(r"\b(angka|bilangan|nilai)\b", re.I)
_NOTE_OPS = [("rata-rata", re.compile(r"rata-?rata|rerata", re.I)), ("jumlah", re.compile(r"\b(jumlah|jumlahkan|total|totalnya)\b", re.I)),
             ("terbesar", re.compile(r"terbesar|tertinggi|maksimum|maksimal|paling\s+besar", re.I)),
             ("terkecil", re.compile(r"terkecil|terendah|minimum|minimal|paling\s+kecil", re.I))]


def notes_tool(q, lex=None, notes=None):
    """Ambil angka dari SATU catatan yang disebut jelas ('catatan 3', 'catatan terakhir') lalu urutkan/jumlahkan/rata-ratakan.
    Angka dari seluruh catatan sengaja tidak digabung (tanggal, telepon, dan nomor lain akan tercampur)."""
    notes = notes or []
    sort = bool(_SORT_VERB.search(q))
    op = "urut" if sort else next((name for name, rx in _NOTE_OPS if rx.search(q)), None)
    ref = _NOTE_REF.search(q)
    if op is None or (not sort and not _NUM_WORD.search(q)):
        return None
    if not ref:
        if _NUM_WORD.search(q) and re.search(r"\b(catatan|memori)\b", q, re.I):
            return {"kind": "task", "tool": None, "detail": "",
                    "answer": "Sebutkan catatan yang dimaksud, mis. 'urutkan angka di catatan 3' atau 'jumlahkan angka di catatan terakhir'. "
                              "Angka dari semua catatan tidak digabung karena tanggal, telepon, dan nomor lain akan ikut tercampur."}
        return None
    if ref.group(1):
        nid = int(ref.group(1))
        note = next((t for i, t in notes if i == nid), None)
    else:
        nid, note = (notes[-1][0], notes[-1][1]) if notes else (None, None)
    name = {"urut": "pengurut angka", "jumlah": "penjumlah angka", "rata-rata": "penghitung rata-rata",
            "terbesar": "pencari angka terbesar", "terkecil": "pencari angka terkecil"}[op]
    if note is None:
        msg = f"Tidak ada catatan aktif dengan nomor {ref.group(1)}." if ref.group(1) else "Belum ada catatan."
        return {"kind": "tool", "tool": name, "answer": msg, "detail": "Hanya catatan yang Anda ajarkan yang bisa dipakai (bukan potongan dokumen)."}
    nums = parse_numbers(note)
    where = f"catatan [{nid}]"
    if not nums:
        return {"kind": "tool", "tool": name, "answer": f"Tidak ada angka di {where}.", "detail": f"Isi: {note}"}
    shown = ", ".join(_format_number(x) for x in nums)
    detail = (f"Angka dibaca dari {where}: {shown}. Periksa bahwa semuanya memang angka yang Anda maksud "
              "(tanggal, jam, dan nomor telepon di catatan ikut terbaca sebagai angka).")
    if op == "urut":
        desc = wants_desc(q)
        ans = ", ".join(_format_number(x) for x in sorted(nums, reverse=desc))
        detail += f" Diurutkan {'menurun (besar ke kecil)' if desc else 'menaik (kecil ke besar)'}."
    elif op == "jumlah":
        ans = _format_number(sum(nums))
    elif op == "rata-rata":
        ans = _format_number(sum(nums) / len(nums))
    else:
        ans = _format_number(max(nums) if op == "terbesar" else min(nums))
    return {"kind": "tool", "tool": name, "answer": ans, "detail": detail}


# ---------- pertanyaan definisi dalam bentuk lain -> "Apa itu X?" ----------
_DEF_NOT_START = QUESTION_WORDS | {"di", "dan", "kalau", "kalo", "untuk", "cara", "proses", "yang", "dengan", "atau", "ke", "dari", "itu", "ini"}
_DEF_FORMS = [
    (re.compile(r"^(?:apa\s+)?(?:pengertian|definisi|arti|artinya|maksud)\s+(?:dari\s+|kata\s+)?(.+?)\s*\??$", re.I), 4),
    (re.compile(r"^(?:jelaskan|terangkan|uraikan|jabarkan)\s+(?:apa\s+itu\s+|tentang\s+|mengenai\s+)?(.+?)\s*\??$", re.I), 3),
    (re.compile(r"^(.+?)\s+(?:itu\s+apa|adalah\s+apa|adalah)\s*\??$", re.I), 4),
]


def rewrite_definition(question):
    """'gravitasi adalah', 'gravitasi itu apa?', 'pengertian gravitasi', 'jelaskan gravitasi' -> 'Apa itu gravitasi?'
    (tanpa tanda tanya: bentuk yang paling dikenal model). Hanya untuk subjek pendek berupa kata benda umum. Kembalikan (pertanyaan, diubah: bool)."""
    q = question.strip()
    if re.match(r"^apa\s+itu\b", q, re.I):
        # model lebih sering menolak 'apa itu X?' (dengan tanda tanya) daripada 'apa itu X'; tanda tanya akhir dibuang
        stripped = q.rstrip("?! ").strip()
        return (stripped, True) if stripped != q else (question, False)
    for rx, max_words in _DEF_FORMS:
        m = rx.match(q)
        if not m:
            continue
        subject = m.group(1).strip(" ,.?!")
        words = subject.split()
        if not words or len(words) > max_words or words[0].lower() in _DEF_NOT_START:
            continue
        if any(w[:1].isupper() for w in words[1:]) or re.search(r"[\d?]", subject):    # nama orang/tempat atau angka: bukan definisi umum
            continue
        return f"Apa itu {subject}", True
    return question, False


# ---------- bentuk pertanyaan alternatif (dicoba hanya bila bentuk pertama tidak menghasilkan jawaban) ----------
_ALT_APA_ITU = re.compile(r"^apa\s+itu\s+(.+?)\s*\??$", re.I)
_ALT_DI_X_MANA = re.compile(r"^di\s+(?!mana\b)(\w+)\s+mana\s+(.+?)\s*\??$", re.I)            # 'di asia mana indonesia berada'
_ALT_REST_DI_X = re.compile(r"^(.+?)\s+di\s+(?!mana\b)(\w+)\s+(?:apa|mana)\s*\??$", re.I)      # 'indonesia berada di kawasan apa'


def alternate_forms(question):
    """Bentuk pertanyaan yang lebih sederhana untuk dicoba bila bentuk pertama gagal. Model hanya terlatih dengan beberapa bentuk
    dan kadang menolak susunan lain; jawaban dari bentuk alternatif tetap melewati penjaga yang sama.
    Kembalikan daftar (bentuk, kata_wajib). kata_wajib = kata penjelas yang dibuang ('asia' pada 'di asia mana ...'): bentuk itu hanya
    boleh dibaca dari catatan yang memuat kata tersebut, agar 'di kota mana ...' tidak dijawab dengan jalan atau jenis lain."""
    q = question.strip()
    m = _ALT_APA_ITU.match(q)
    if m:
        return [(f"Apa {m.group(1)}?", None)]
    m = _ALT_DI_X_MANA.match(q)
    if m:
        return [(f"Di mana {m.group(2)}?", m.group(1))]
    m = _ALT_REST_DI_X.match(q)
    if m:
        return [(f"{m.group(1)} di mana", m.group(2))]
    return []


# ---------- perutean ----------
def is_task(q):
    return bool(re.match(r"^\s*(tolong\s+|coba\s+)?" + TASK_VERBS + r"\b", q, re.I))


def route(q, lex=None, notes=None):
    """Kembalikan hasil alat, pesan 'permintaan tugas', atau None bila ini pertanyaan/pernyataan biasa.
    notes: daftar (id, teks) catatan pengguna, untuk alat yang mengambil angka dari catatan."""
    r = notes_tool(q, lex, notes)
    if r:
        return r
    for fn in (sort_tool, date_tool, calc_tool, terbilang_tool):
        r = fn(q, lex) if fn is not date_tool else fn(q)
        if r:
            return r
    if is_task(q):
        return {"kind": "task", "tool": None, "answer": TASK_MESSAGE, "detail": ""}
    return None
