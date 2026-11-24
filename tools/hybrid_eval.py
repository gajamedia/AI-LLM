"""
Ukur pencarian: BM25 murni vs vektor n-gram vs hibrida (RRF). Hanya mengukur PENCARIAN (apakah bagian yang benar
ada di k teratas), bukan jawaban model. Semua soal berasal dari set asli/tahan-uji; tidak ada data karangan baru.

  A) Paragraf Wikipedia (TyDi QA / IDK-MRC, set tahan uji data/sft/sft_qa_eval.jsonl) di antara ribuan paragraf pengganggu
     dari data latih QA. Cocok dengan kasus "impor dokumen"/RAG.
  B) Catatan pendek (evalsets/memory_*.jsonl): semua catatan dijadikan satu memori.
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hybrid_search as H
import rag_lib as R

CTX = re.compile(r"Bacalah teks berikut\.\n(.*)\nBerdasarkan teks tersebut", re.S)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def bm25_ranker(texts):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE f USING fts5(text)")
    con.executemany("INSERT INTO f(rowid, text) VALUES (?,?)", [(i, t) for i, t in enumerate(texts)])

    def rank(q, k=50):
        qq = R.build_query(q)
        if not qq:
            return []
        return [r[0] for r in con.execute("SELECT rowid FROM f WHERE f MATCH ? ORDER BY bm25(f) LIMIT ?", (qq, k))]
    return rank


def metrics(rank_lists, golds):
    n = len(golds)
    h1 = h3 = h10 = 0
    mrr = 0.0
    for r, g in zip(rank_lists, golds):
        pos = next((i for i, x in enumerate(r) if x in g), None)
        if pos is not None:
            h1 += pos < 1; h3 += pos < 3; h10 += pos < 10
            mrr += 1.0 / (pos + 1)
    return {"n": n, "hit@1": round(100 * h1 / n, 1), "hit@3": round(100 * h3 / n, 1),
            "hit@10": round(100 * h10 / n, 1), "mrr": round(mrr / n, 3)}


def run(name, texts, queries, golds, configs, embedder=None):
    bm = bm25_ranker(texts)
    out = {"bm25": metrics([bm(q) for q in queries], golds)}
    ids = list(range(len(texts)))
    for label, kw in configs.items():
        t0 = time.time()
        vi = H.VectorIndex(**kw).fit(ids, texts)
        vec_lists = [[i for i, _ in vi.search(q, 50)] for q in queries]
        out["vektor " + label] = metrics(vec_lists, golds)
        out["hibrida " + label] = metrics([H.rrf([bm(q), v]) for q, v in zip(queries, vec_lists)], golds)
        out["hibrida " + label]["dtk_bangun+cari"] = round(time.time() - t0, 1)
    if embedder is not None:
        import numpy as np
        P = embedder.encode_passages(texts)
        vec_lists = [[int(i) for i in np.argsort(-(P @ embedder.encode_query(q)))[:50]] for q in queries]
        out["vektor " + embedder.name.split("/")[-1]] = metrics(vec_lists, golds)
        out["hibrida " + embedder.name.split("/")[-1]] = metrics([H.rrf([bm(q), v]) for q, v in zip(queries, vec_lists)], golds)
    print(f"\n== {name}: {len(texts)} bagian, {len(queries)} soal ==")
    print(f"{'metode':<28}{'hit@1':>7}{'hit@3':>7}{'hit@10':>8}{'mrr':>7}")
    for k, m in out.items():
        print(f"{k:<28}{m['hit@1']:>7}{m['hit@3']:>7}{m['hit@10']:>8}{m['mrr']:>7}")
    return out


def bench_wiki(n_distract, configs, embedder=None):
    ev = [json.loads(l) for l in open(ROOT / "data/sft/sft_qa_eval.jsonl", encoding="utf-8")]
    ev = [r for r in ev if r.get("answerable")]
    gold_ctx = sorted({norm(r["context"]) for r in ev})
    pool, seen = [], set()
    for r in ev:
        k = norm(r["context"])
        if k not in seen:
            seen.add(k); pool.append(r["context"])
    for l in open(ROOT / "data/sft/sft_qa.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d.get("source") not in ("qa_tydi", "qa_idkmrc"):
            continue
        m = CTX.search(d["prompt"])
        if m and norm(m.group(1)) not in seen:
            seen.add(norm(m.group(1))); pool.append(m.group(1))
        if len(pool) >= len(gold_ctx) + n_distract:
            break
    index = {norm(t): i for i, t in enumerate(pool)}
    return run("A) Wikipedia", pool, [r["question"] for r in ev], [{index[norm(r["context"])]} for r in ev], configs, embedder)


def bench_notes(configs, embedder=None):
    rows = []
    for f in ("memory_complex", "memory_handwritten"):
        rows += [json.loads(l) for l in open(ROOT / f"evalsets/{f}.jsonl", encoding="utf-8")]
    for f in ("memory_cases", "memory_cases_v7_segar"):
        for l in open(ROOT / f"evalsets/{f}.jsonl", encoding="utf-8"):
            d = json.loads(l)
            if not d.get("answer"):
                continue
            for n in d["notes"]:
                rows.append({"note": n, "question": d["question"], "answer": d["answer"], "multi": True})
    pool, seen = [], {}
    for r in rows:
        if r["note"] not in seen:
            seen[r["note"]] = len(pool); pool.append(r["note"])
    queries, golds = [], []
    for r in rows:
        if r.get("multi") and norm(r["answer"]) not in norm(r["note"]):
            continue   # pada set multi-catatan, hanya catatan yang memuat jawaban yang benar
        queries.append(r["question"]); golds.append({seen[r["note"]]})
    merged = {}
    for q, g in zip(queries, golds):
        merged.setdefault(q, set()).update(g)
    return run("B) Catatan pendek", pool, list(merged), list(merged.values()), configs, embedder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractors", type=int, default=3500)
    ap.add_argument("--embed", default=None, help="nama model embedding gaya E5 (butuh torch+transformers), mis. intfloat/multilingual-e5-small")
    ap.add_argument("--out", default="reports/eval_hybrid_search.json")
    args = ap.parse_args()
    configs = {"n-gram": dict(dim=2048, lsa_dim=0), "LSA256": dict(dim=2048, lsa_dim=256)}
    emb = H.E5Embedder(args.embed) if args.embed else None
    res = {"wiki": bench_wiki(args.distractors, configs, emb), "catatan": bench_notes({"n-gram": dict(dim=2048, lsa_dim=0)}, emb)}
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nditulis:", args.out)


if __name__ == "__main__":
    main()
