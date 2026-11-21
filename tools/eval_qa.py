"""
Evaluasi QA berbasis teks pada set held-out (data/sft/sft_qa_eval.jsonl).

Decoding greedy tanpa repetition penalty (penalti merusak penyalinan dari konteks).
Metrik:
  - terjawab:     contains (jawaban emas ada di keluaran) dan token-F1
  - tak terjawab: tingkat penolakan (model menyatakan tidak ada di teks)
  - false_refusal: model menolak padahal jawabannya ada
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from config import get_config
from generate_gpt import _load_checkpoint_payload, _load_model, _set_seed, generate_text

TEMPLATE_PROMPT = "Pertanyaan: {p}\nJawaban:"
REFUSE = re.compile(r"tidak\s+(ada|terdapat|menyebut|disebut|dijelaskan|tersedia)|tidak diketahui|maaf", re.I)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env", default="production_100m")
    ap.add_argument("--eval-file", default="data/sft/sft_qa_eval.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cpu")
    cfg = get_config(args.env)
    ckpt = Path(args.checkpoint)
    payload = _load_checkpoint_payload(ckpt)
    model, tok = _load_model(ckpt, cfg, device, payload=payload)
    del payload

    rows = [json.loads(l) for l in open(args.eval_file, encoding="utf-8")][:args.n]
    results, ans_hit, ans_f1, ans_ref, ans_n, un_ref, un_n = [], 0, 0.0, 0, 0, 0, 0
    for i, r in enumerate(rows):
        prompt = TEMPLATE_PROMPT.format(p=r["prompt"])
        _set_seed(1)
        text = generate_text(model, tok, prompt, args.max_new_tokens, 0.0, 0, 1.0, 1.0, device)
        # decode() dapat mengubah baris baru menjadi spasi; potong sesudah penanda "Jawaban:" milik template.
        k = text.find("Jawaban:", max(0, len(prompt) - 20))
        out = (text[k + len("Jawaban:"):] if k >= 0 else text[len(prompt):]).strip()
        refused = bool(REFUSE.search(out))
        if r["answerable"]:
            ans_n += 1
            hit = norm(r["response"]) in norm(out)
            ans_hit += hit
            ans_f1 += f1(out, r["response"])
            ans_ref += refused
        else:
            un_n += 1
            un_ref += refused
            hit = None
        results.append({"q": r["question"], "gold": r["response"], "answerable": r["answerable"],
                        "out": out, "refused": refused, "hit": hit})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    summary = {
        "checkpoint": str(ckpt), "step": None, "n": len(rows),
        "answerable_n": ans_n, "contains_acc": ans_hit / max(ans_n, 1), "token_f1": ans_f1 / max(ans_n, 1),
        "false_refusal_rate": ans_ref / max(ans_n, 1),
        "unanswerable_n": un_n, "refusal_rate_on_unanswerable": un_ref / max(un_n, 1),
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
