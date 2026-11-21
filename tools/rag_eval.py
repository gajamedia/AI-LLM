"""
Evaluasi RAG ujung ke ujung: pertanyaan -> cari di indeks FTS5 (BM25) -> model menjawab dari potongan.

Hanya soal terjawab dari set held-out (sft_qa_eval.jsonl) yang dinilai; jawaban emas dianggap benar
bila ada di keluaran (setelah normalisasi). Strategi pembaca:
  top1       : pakai potongan peringkat 1
  sequential : coba potongan 1..K berurutan, ambil jawaban pertama yang BUKAN penolakan (kosong = penolakan)
  oracle     : konteks emas (batas atas pembaca; sama dengan eval_qa.py)
Potongan dipotong ke jendela kalimat yang paling cocok dengan pertanyaan dan muat dalam batas token.
Pencarian dinilai sendiri: recall@k = jawaban emas ada di salah satu potongan k teratas.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import rag_lib as R
from config import get_config
from generate_gpt import _load_checkpoint_payload, _load_model, _set_seed, generate_text


def read(model, tok, ctx, question, device, max_new):
    prompt = R.TEMPLATE_PROMPT.format(p=R.make_prompt(ctx, question))
    _set_seed(1)
    text = generate_text(model, tok, prompt, max_new, 0.0, 0, 1.0, 1.0, device)
    k = text.find("Jawaban:", max(0, len(prompt) - 20))
    return (text[k + len("Jawaban:"):] if k >= 0 else text[len(prompt):]).strip()


def is_refusal(a):
    return (not a.strip()) or bool(R.REFUSE.search(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="reports/checkpoints_100m_sft_qa/gpt_sft_best.pt")
    ap.add_argument("--db", default="data/rag/wiki_fts.db")
    ap.add_argument("--eval-file", default="data/sft/sft_qa_eval.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--recall-k", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--out", default="reports/eval_rag.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    cfg = get_config("production_100m")
    ckpt = Path(args.checkpoint)
    payload = _load_checkpoint_payload(ckpt)
    model, tok = _load_model(ckpt, cfg, device, payload=payload)
    del payload

    con = R.open_db(args.db)
    rows = [json.loads(l) for l in open(args.eval_file, encoding="utf-8")][:args.n]
    rows = [r for r in rows if r["answerable"]]
    n = len(rows)

    # --- 1) pencarian ---
    ret, rec = [], {1: 0, 3: 0, 5: 0, args.recall_k: 0}
    for r in rows:
        hits = R.retrieve(con, r["question"], args.recall_k)
        gold = R.norm(r["response"])
        pos = next((i + 1 for i, h in enumerate(hits) if gold in R.norm(h["text"])), None)
        for k in rec:
            rec[k] += int(pos is not None and pos <= k)
        for h in hits:
            h["win"] = None  # dihitung malas saat dibaca
        ret.append(hits)
    print("== pencarian (n=%d soal terjawab, indeks %s)" % (n, args.db), flush=True)
    for k in sorted(rec):
        print(f"  recall@{k}: {rec[k]/n:.3f}", flush=True)

    # --- 2) pembaca ---
    stats = {"top1": 0, "sequential": 0, "oracle": 0}
    win_has_gold = 0
    out_rows = []
    for i, r in enumerate(rows):
        gold = R.norm(r["response"])
        q = r["question"]
        hits = ret[i][:args.k]
        answers = []
        for h in hits:
            h["win"] = R.fit_window(tok, h["text"], q)
            answers.append(read(model, tok, h["win"], q, device, args.max_new_tokens) if h["win"] else "")
        a_top1 = answers[0] if answers else ""
        a_seq, used = "Informasi tersebut tidak ada dalam teks.", None
        for j, a in enumerate(answers):
            if not is_refusal(a):
                a_seq, used = a, j + 1
                break
        a_or = read(model, tok, r["context"], q, device, args.max_new_tokens)
        ok = {"top1": gold in R.norm(a_top1) and not is_refusal(a_top1),
              "sequential": gold in R.norm(a_seq) and used is not None,
              "oracle": gold in R.norm(a_or)}
        for k_, v in ok.items():
            stats[k_] += int(v)
        has = [bool(h["win"]) and gold in R.norm(h["win"]) for h in hits]
        win_has_gold += int(any(has))
        out_rows.append({"q": q, "gold": r["response"], "top1": a_top1, "sequential": a_seq, "used_rank": used,
                         "oracle": a_or, "window_has_gold": has, **{f"ok_{k_}": v for k_, v in ok.items()}})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n} " + " ".join(f"{k_}={v/(i+1):.3f}" for k_, v in stats.items()), flush=True)

    summary = {"n": n, "retrieval_recall": {f"@{k}": rec[k] / n for k in sorted(rec)},
               f"window_has_gold@{args.k}": win_has_gold / n,
               "reader_acc": {k_: v / n for k_, v in stats.items()}}
    print(json.dumps(summary, indent=1))
    Path(args.out).write_text(json.dumps({"summary": summary, "results": out_rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
