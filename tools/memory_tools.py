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

TOOL_LIST = "mengurutkan angka, menghitung (+ - x / ^), dan hari/selisih tanggal"
TASK_MESSAGE = ("Ini permintaan tugas, bukan pertanyaan tentang catatan Anda. Aplikasi ini hanya membaca catatan yang "
                "Anda ajarkan, jadi tidak bisa menulis, menerjemahkan, atau merangkum. Alat yang tersedia: " + TOOL_LIST + ".")

MONTHS = {"januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
          "september": 9, "oktober": 10, "november": 11, "desember": 12}
DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
TASK_VERBS = r"(urutkan|mengurutkan|sortir|hitung|hitunglah|jumlahkan|kalikan|terjemahkan|tuliskan|buatkan|ringkaskan|rangkum|ceritakan|translate)"


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


_UNITS = {"nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9}
_NUMBER_WORDS = list(_UNITS) + ["sepuluh", "sebelas", "belas", "puluh", "seratus", "ratus", "seribu", "ribu"]
_RUN = re.compile(r"\b(?:" + "|".join(_NUMBER_WORDS) + r")\b(?:\s+(?:" + "|".join(_NUMBER_WORDS) + r")\b)*", re.I)


def _parse_number_run(tokens):
    """Ubah deret kata bilangan (mis. 'dua puluh lima', 'seratus tiga belas', 'tiga satu dua') menjadi daftar bilangan bulat."""
    nums, total, sub, cur, last = [], 0, 0, None, None

    def flush():
        nonlocal total, sub, cur, last
        if last is not None:
            nums.append(total + sub + (cur or 0))
        total, sub, cur, last = 0, 0, None, None

    for t in tokens:
        if t in _UNITS:
            if last in ("unit", "teen"):
                flush()
            cur, last = _UNITS[t], "unit"
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
        elif t == "seribu":
            if last is not None:
                flush()
            total, last = 1000, "thousand"
        elif t == "ribu" and last in ("unit", "teen", "tens", "hundreds"):
            total, sub, cur, last = (sub + (cur or 0)) * 1000, 0, None, "thousand"
    flush()
    return nums


def words_to_digits(text):
    """Ganti kata bilangan Indonesia dengan angka: 'dua puluh lima ditambah satu' -> '25 ditambah 1'."""
    def repl(m):
        nums = _parse_number_run(m.group(0).lower().split())
        return " ".join(str(n) for n in nums) if nums else m.group(0)
    return _RUN.sub(repl, text)


def sort_tool(q):
    if not re.search(r"\b(urutkan|mengurutkan|sortir|sort)\b", q, re.I):
        return None
    nums = parse_numbers(words_to_digits(q))
    if len(nums) < 2:
        return None
    desc = bool(re.search(r"terbesar|besar\s*(?:ke|->|-)\s*kecil|menurun|turun|descending|dari yang besar", q, re.I))
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


_WORD_OPS = [(r"\bditambah(?:kan)?\b|\btambah\b|\bplus\b", " + "), (r"\bdikurangi\b|\bminus\b", " - "),
             (r"\bdikali(?:kan)?\b|\bkali\b", " * "), (r"\bdibagi\b", " / "), (r"\bpangkat\b", " ** ")]
_FILLER = {"berapa", "berapakah", "hitung", "hitunglah", "hasil", "hasilnya", "dari", "adalah", "ini", "itu", "tolong",
           "yang", "ya", "saja", "dong", "sama", "dengan", "nya", "jumlah", "jumlahkan", "kalikan", "bagi", "kurang", "kurangi"}
_TRIGGER = re.compile(r"\b(hitung|hitunglah|berapa|berapakah|hasil|jumlahkan|kalikan)\b", re.I)


def calc_tool(q):
    t = words_to_digits(q.lower()).strip().rstrip("?= ")
    if re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", t) or re.search(r"\b\d{3,}-\d{3,}", t):
        return None  # tanggal atau nomor telepon, bukan pengurangan
    expr_text = t
    for pat, rep in _WORD_OPS:
        expr_text = re.sub(pat, rep, expr_text)
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
    if len(ds) == 1 and re.search(r"\bhari apa\b", q, re.I):
        d = ds[0]
        return {"kind": "tool", "tool": "tanggal", "answer": DAYS[d.weekday()],
                "detail": f"{d.day} {[k for k, v in MONTHS.items() if v == d.month][0].capitalize()} {d.year} jatuh pada hari {DAYS[d.weekday()]}."}
    if len(ds) == 2 and re.search(r"\b(berapa hari|selisih|jarak)\b", q, re.I):
        n = abs((ds[1] - ds[0]).days)
        return {"kind": "tool", "tool": "tanggal", "answer": f"{n} hari", "detail": f"Selisih antara {ds[0].isoformat()} dan {ds[1].isoformat()}."}
    return None


# ---------- sapaan dan saran mengajar (dipakai CLI dan web) ----------
INTRO = ("AI ini hanya tahu apa yang Anda ajarkan; ia tidak menjawab dari pengetahuan umum.\n"
         "Mulai dengan:  ingat: Bu Rina tinggal di Jalan Mawar nomor 5.\n"
         "Lalu tanya:    Di mana Bu Rina tinggal?")
GREETING = re.compile(
    r"^(?:(?:halo|hai|hi|hei|hello|selamat\s+(?:pagi|siang|sore|malam))(?:\s+(?:apa\s+kabar|semua|semuanya|teman|sobat|kak|kakak|pak|bu|mas|mbak|bro))?"
    r"|assalamu\S*|apa\s+kabar|terima\s+kasih(?:\s+banyak)?|makasih|thanks?)\W*$", re.I)
TABLE_HINT = "Tabel atau rumus tidak bisa dipelajari dari catatan. Untuk menghitung, ketik langsung, mis. 'hitung 12 x 5'."


def greeting_reply(n_notes):
    base = "Halo! Saya memori pribadi Anda: saya hanya tahu apa yang Anda ajarkan, bukan pengetahuan umum.\n"
    if n_notes == 0:
        return base + INTRO
    return base + f"Saat ini ada {n_notes} catatan. Tanyakan dengan kalimat berakhiran '?', atau tambah catatan dengan 'ingat: ...'."


def looks_like_table(text):
    """Catatan berisi tabel/rumus hitung (mis. '1+1=2 1+2=3 ...'), yang tidak bisa dipelajari model."""
    return len(re.findall(r"\d\s*[+\-x×*/]\s*\d\s*=\s*\d", text)) >= 2 or text.count("=") >= 3


# ---------- perutean ----------
def is_task(q):
    return bool(re.match(r"^\s*(tolong\s+|coba\s+)?" + TASK_VERBS + r"\b", q, re.I))


def route(q):
    """Kembalikan hasil alat, pesan 'permintaan tugas', atau None bila ini pertanyaan/pernyataan biasa."""
    for fn in (sort_tool, date_tool, calc_tool):
        r = fn(q)
        if r:
            return r
    if is_task(q):
        return {"kind": "task", "tool": None, "answer": TASK_MESSAGE, "detail": ""}
    return None
