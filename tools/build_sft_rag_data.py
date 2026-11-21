"""
SFT tahap 3: latih pembaca pada hasil pencarian nyata (BM25 di indeks Wikipedia).

Untuk tiap pertanyaan latih yang terjawab (data/sft/sft_qa.jsonl, sumber qa_*), cari top-N potongan:
  - positif : potongan peringkat terbaik yang jendelanya memuat jawaban -> jawab dengan kutipan
  - negatif : potongan teratas yang TIDAK memuat jawaban (dan tumpang-tindih token jawaban < 0.5)
              -> jawab "tidak ada dalam teks" (negatif keras: terlihat relevan tetapi meleset)
Negatif dibatasi --neg-ratio x jumlah positif agar model tidak terlalu sering menolak.
Ditambah replay contoh QA bersih, dan contoh instruksi umum.
Soal tak terjawab (IDK-MRC) tidak dipakai untuk pencarian: jawabannya mungkin ada di tempat lain di Wikipedia.
"""
import argparse
import json
import multiprocessing as mp
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import rag_lib as R

_TOK = None
_CON = None
_ARGS = None


def _init(db, topn):
    global _TOK, _CON, _ARGS
    from config import get_config
    from embedding import build_tokenizer_and_embedding
    _TOK, _ = build_tokenizer_and_embedding(cfg=get_config("production_100m"), load_tokenizer=True)
    _CON = R.open_db(db)
    _ARGS = topn


def _coverage(answer, passage_norm):
    toks = R.norm(answer).split()
    return sum(1 for t in toks if t in passage_norm.split()) / max(len(toks), 1)


def _work(item):
    q, ans = item["question"], item["response"]
    gold = R.norm(ans)
    hits = R.retrieve(_CON, q, _ARGS, exclude_src=("extra",))
    pos, negs = None, []
    for rank, h in enumerate(hits, 1):
        win = R.fit_window(_TOK, h["text"], q)
        if not win:
            continue
        full_n = R.norm(h["text"])
        if gold in R.norm(win):
            if pos is None:
                pos = {"rank": rank, "ctx": win}
        elif gold not in full_n and _coverage(ans, full_n) < 0.5 and len(negs) < 2:
            negs.append({"rank": rank, "ctx": win})
    out = {"q": q, "ans": ans, "pos": pos, "negs": negs, "found": any(gold in R.norm(h["text"]) for h in hits)}
    return out


def count_tokens(tok, prompt, resp):
    return len(tok.encode_ids(R.TEMPLATE_PROMPT.format(p=prompt) + " " + resp, add_bos=True, add_eos=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/sft/sft_qa.jsonl")
    ap.add_argument("--db", default="data/rag/wiki_fts.db")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--neg-ratio", type=float, default=0.7)
    ap.add_argument("--val-questions", type=int, default=300)
    ap.add_argument("--replay-clean-ans", type=int, default=4000)
    ap.add_argument("--replay-clean-unans", type=int, default=1500)
    ap.add_argument("--replay-general", type=int, default=5000)
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/sft")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in open(args.qa, encoding="utf-8")]
    qa = [r for r in rows if r["source"].startswith("qa_")]
    answerable = [r for r in qa if r["answerable"]]
    clean_unans = [r for r in qa if not r["answerable"]]
    general = [r for r in rows if not r["source"].startswith("qa_")]
    rng.shuffle(answerable)
    if args.max_questions:
        answerable = answerable[:args.max_questions]
    print(f"[rag-sft] pertanyaan terjawab={len(answerable)} (QA bersih tak terjawab={len(clean_unans)}, umum={len(general)})", flush=True)

    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.db, args.topn)) as pool:
        results = []
        for i, r in enumerate(pool.imap(_work, answerable, chunksize=16), 1):
            results.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(answerable)} pertanyaan, {time.time()-t0:.0f}s", flush=True)
    found = sum(r["found"] for r in results)
    with_pos = sum(1 for r in results if r["pos"])
    print(f"[rag-sft] jawaban ada di top-{args.topn}: {found}/{len(results)}; punya positif: {with_pos}", flush=True)

    from config import get_config
    from embedding import build_tokenizer_and_embedding
    tok, _ = build_tokenizer_and_embedding(cfg=get_config("production_100m"), load_tokenizer=True)

    def example(ctx, q, resp, source, rank=None):
        prompt = R.make_prompt(ctx, q)
        n = count_tokens(tok, prompt, resp)
        return {"prompt": prompt, "response": resp, "source": source, "n_tokens": n, "rank": rank, "question": q}

    split = args.val_questions
    stats = Counter()
    sets = {"val": results[:split], "train": results[split:]}
    built = {}
    for name, rs in sets.items():
        pos_ex, neg_ex = [], []
        for r in rs:
            if r["pos"]:
                pos_ex.append(example(r["pos"]["ctx"], r["q"], r["ans"], "rag_pos", r["pos"]["rank"]))
            for ng in r["negs"]:
                neg_ex.append(example(ng["ctx"], r["q"], rng.choice(R.NO_ANSWER), "rag_neg", ng["rank"]))
        pos_ex = [e for e in pos_ex if e["n_tokens"] <= R.MAX_CONTEXT]
        neg_ex = [e for e in neg_ex if e["n_tokens"] <= R.MAX_CONTEXT]
        rng.shuffle(neg_ex)
        neg_ex = neg_ex[:int(args.neg_ratio * len(pos_ex))]
        stats[f"{name}:rag_pos"], stats[f"{name}:rag_neg"] = len(pos_ex), len(neg_ex)
        built[name] = pos_ex + neg_ex

    clean_ans = [r for r in answerable[split:]]
    rng.shuffle(clean_ans)
    replay = clean_ans[:args.replay_clean_ans]
    rng.shuffle(clean_unans)
    replay += clean_unans[:args.replay_clean_unans]
    rng.shuffle(general)
    replay += general[:args.replay_general]
    for r in replay:
        r.pop("context", None)

    train = built["train"] + replay
    rng.shuffle(train)
    val = built["val"] + [json.loads(l) for l in open("data/sft/sft_qa_val.jsonl", encoding="utf-8")][:300]

    out = Path(args.out_dir)
    for path, rs in ((out / "sft_rag.jsonl", train), (out / "sft_rag_val.jsonl", val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(stats), indent=1))
    print(f"[rag-sft] latih={len(train)} (rag_pos={stats['train:rag_pos']}, rag_neg={stats['train:rag_neg']}, "
          f"replay={len(replay)}) | validasi={len(val)}")
    rk = Counter(e["rank"] for e in built["train"] if e["source"] == "rag_pos")
    print("peringkat positif:", dict(sorted(rk.items())))


if __name__ == "__main__":
    mp.freeze_support()
    main()
