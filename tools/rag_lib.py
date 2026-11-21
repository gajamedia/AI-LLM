"""
Pustaka bersama RAG: pembuatan kueri, pencarian FTS5 (BM25), dan pemilihan jendela konteks
yang muat dalam batas token model (konteks 256).
"""
import re
import sqlite3

TEMPLATE_PROMPT = "Pertanyaan: {p}\nJawaban:"
QA_PROMPT = "Bacalah teks berikut.\n{ctx}\nBerdasarkan teks tersebut, jawab: {q}"
NO_ANSWER = [
    "Informasi tersebut tidak ada dalam teks.",
    "Teks tidak menyebutkan hal itu.",
    "Maaf, jawabannya tidak terdapat dalam teks yang diberikan.",
]
REFUSE = re.compile(r"tidak\s+(ada|terdapat|menyebut|disebut|dijelaskan|tersedia)|tidak diketahui|maaf", re.I)
STOP = set("""yang di ke dari dan atau apa apakah siapa kapan dimana dimanakah mana manakah bagaimana berapa mengapa kenapa
adalah ialah itu ini untuk dengan pada dalam oleh sebagai kah nya pun akan telah sudah dapat bisa para saja lah
tahun terjadi disebut dinamakan bernama yaitu yakni tersebut dia ia mereka kita kami saya anda ada tidak bukan
kapankah siapakah apakah bagaimanakah berapakah mengapakah dimanakah kemanakah darimanakah dimaksud""".split())
MAX_CONTEXT = 256
GEN_MARGIN = 40


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def content_tokens(question):
    out, seen = [], set()
    for t in re.findall(r"\w+", question.lower()):
        if t in STOP or len(t) < 2 or t in seen:
            continue
        seen.add(t); out.append(t)
    return out


def build_query(question, max_tokens=14):
    return " OR ".join('"%s"' % t for t in content_tokens(question)[:max_tokens])


def open_db(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def retrieve(con, question, k, exclude_src=()):
    q = build_query(question)
    if not q:
        return []
    rows = con.execute(
        "SELECT title, text, src FROM passages WHERE passages MATCH ? ORDER BY bm25(passages, 3.0, 1.0) LIMIT ?",
        (q, k + 5)).fetchall()
    rows = [r for r in rows if r[2] not in exclude_src][:k]
    return [{"title": t, "text": x, "src": s} for t, x, s in rows]


def _sentences(text):
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return parts or [text]


def overhead_tokens(tokenizer, question):
    prompt = TEMPLATE_PROMPT.format(p=QA_PROMPT.format(ctx="", q=question))
    return len(tokenizer.encode_ids(prompt, add_bos=True, add_eos=True))


def fit_window(tokenizer, passage, question, margin=GEN_MARGIN):
    """Jendela kalimat berurutan dari passage yang paling cocok dengan pertanyaan dan muat dalam token."""
    budget = MAX_CONTEXT - overhead_tokens(tokenizer, question) - margin
    if budget < 20:
        return None
    count = lambda s: len(tokenizer.encode_ids(s))
    if count(passage) <= budget:
        return passage
    sents = _sentences(passage)
    qt = set(content_tokens(question))
    scores = [len(qt & set(re.findall(r"\w+", s.lower()))) for s in sents]
    center = max(range(len(sents)), key=lambda i: (scores[i], -i))
    lens = [count(s) + 1 for s in sents]
    lo = hi = center
    used = lens[center]
    if used > budget:  # satu kalimat kepanjangan: potong per karakter
        s, a, b = sents[center], 0, len(sents[center])
        while a < b:
            mid = (a + b + 1) // 2
            if count(s[:mid]) <= budget:
                a = mid
            else:
                b = mid - 1
        return s[:a]
    while True:
        grew = False
        for side in ("r", "l"):
            if side == "r" and hi + 1 < len(sents) and used + lens[hi + 1] <= budget:
                hi += 1; used += lens[hi]; grew = True
            elif side == "l" and lo - 1 >= 0 and used + lens[lo - 1] <= budget:
                lo -= 1; used += lens[lo]; grew = True
        if not grew:
            break
    return " ".join(sents[lo:hi + 1])


def make_prompt(ctx, question):
    return QA_PROMPT.format(ctx=ctx, q=question)
