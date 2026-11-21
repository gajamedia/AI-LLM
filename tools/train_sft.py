"""
SFT (supervised fine-tuning) untuk GPT: loss hanya pada token jawaban.

Format contoh:  <BOS> "Pertanyaan: {prompt}\\nJawaban:" " {jawaban}" <EOS>
Label prompt = -100 (tidak dihitung). Batch dikelompokkan menurut panjang agar padding sedikit.
Mulai dari bobot pretraining via --init-from (optimizer baru, LR kecil).
"""
import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from checkpoint_meta import validate_checkpoint_metadata
from config import get_config
from gptModel import GPTModel
from train_gptV2 import _optimizer_step, _resolve_device, _save_checkpoint, _warmup_cosine_factor

TEMPLATE_PROMPT = "Pertanyaan: {p}\nJawaban:"

_TOK = None


def _worker_init(env):
    global _TOK
    from embedding import build_tokenizer_and_embedding
    _TOK, _ = build_tokenizer_and_embedding(cfg=get_config(env), load_tokenizer=True)


def _encode_example(row):
    """Kembalikan (ids, n_prompt). Prompt di-encode terpisah; jika prefiksnya tidak sama
    dengan encoding gabungan, jawaban di-encode terpisah dan disambung (konsisten dengan inferensi)."""
    tok = _TOK
    p_text = TEMPLATE_PROMPT.format(p=row["prompt"])
    p_ids = tok.encode_ids(p_text, add_bos=True)
    full = tok.encode_ids(p_text + " " + row["response"], add_bos=True, add_eos=True)
    if full[:len(p_ids)] == p_ids:
        return full, len(p_ids)
    r_ids = tok.encode_ids(row["response"], add_eos=True)
    return p_ids + r_ids, len(p_ids)


def load_examples(path, env, cache, max_len, max_examples, workers):
    cache = Path(cache) if cache else None
    if cache and cache.exists():
        data = torch.load(cache)
        print(f"[sft] cache token dimuat: {cache} ({len(data)} contoh)")
    else:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        t0 = time.time()
        with mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(env,)) as pool:
            data = pool.map(_encode_example, rows, chunksize=256)
        print(f"[sft] tokenisasi {len(rows)} contoh dalam {time.time()-t0:.0f}s")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, cache)
    kept = [(ids, n) for ids, n in data if len(ids) <= max_len and n < len(ids)]
    print(f"[sft] dipakai {len(kept)}/{len(data)} contoh (<= {max_len} token)")
    return kept[:max_examples] if max_examples else kept


def make_batches(examples, batch_size, shuffle, seed):
    """Kelompokkan contoh berurutan menurut panjang dalam 'megabatch' agar padding kecil."""
    idx = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    mega = batch_size * 50
    batches = []
    for s in range(0, len(idx), mega):
        chunk = sorted(idx[s:s + mega], key=lambda i: len(examples[i][0]))
        batches += [chunk[b:b + batch_size] for b in range(0, len(chunk), batch_size)]
    if shuffle:
        random.Random(seed + 1).shuffle(batches)
    return batches


def collate(examples, batch_idx, pad_id, device):
    seqs = [examples[i] for i in batch_idx]
    T = max(len(ids) for ids, _ in seqs) - 1
    inp = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), T), -100, dtype=torch.long)
    for r, (ids, n_prompt) in enumerate(seqs):
        t = torch.tensor(ids, dtype=torch.long)
        inp[r, :len(ids) - 1] = t[:-1]
        tgt = t[1:].clone()
        tgt[:n_prompt - 1] = -100          # posisi yang memprediksi token prompt
        lab[r, :len(ids) - 1] = tgt
    return {"input_ids": inp.to(device), "labels": lab.to(device)}


@torch.inference_mode()
def evaluate(model, examples, pad_id, device, batch_size, max_batches, use_amp, amp_dtype):
    model.eval()
    tot, cnt = 0.0, 0
    for k, b in enumerate(make_batches(examples, batch_size, False, 0)):
        if max_batches and k >= max_batches:
            break
        batch = collate(examples, b, pad_id, device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss, _ = model.forward_batch(batch)
        n = int((batch["labels"] != -100).sum())
        tot += float(loss) * n
        cnt += n
    model.train()
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="production_100m")
    ap.add_argument("--init-from", required=True, help="Checkpoint pretraining (gpt_best.pt).")
    ap.add_argument("--data", default="data/sft/sft_id.jsonl")
    ap.add_argument("--val-data", default="data/sft/sft_id_val.jsonl")
    ap.add_argument("--token-cache", default="data/sft/sft_tokens_train.pt")
    ap.add_argument("--val-token-cache", default="data/sft/sft_tokens_val.pt")
    ap.add_argument("--save-dir", default="reports/checkpoints_100m_sft")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="Batasi total optimizer step (0 = sesuai epochs).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=0, help="0 = seluruh set validasi.")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--tok-workers", type=int, default=6)
    ap.add_argument("--num-threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenize-only", action="store_true", help="Hanya buat cache token lalu keluar.")
    args = ap.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    cfg = get_config(args.env)

    train_ex = load_examples(args.data, args.env, args.token_cache, args.max_len, args.max_examples, args.tok_workers)
    val_ex = load_examples(args.val_data, args.env, args.val_token_cache, args.max_len, 0, args.tok_workers)

    if args.tokenize_only:
        print("[sft] tokenize-only selesai.")
        return

    model, tokenizer = GPTModel.from_config_with_tokenizer(cfg=cfg, load_tokenizer=True)
    payload = torch.load(args.init_from, map_location="cpu")
    validate_checkpoint_metadata(payload=payload, model=model, tokenizer=tokenizer, cfg=cfg, strict=False)
    model.load_compatible_state_dict(payload["model_state_dict"], strict=True)
    print(f"[sft] bobot dimuat dari {args.init_from} (pretrain step={payload.get('step')})")
    del payload
    model.to(device).train()
    pad_id = int(tokenizer.token2id[tokenizer.pad_token])

    use_amp = bool(args.amp) and device.type in ("cuda", "cpu")
    amp_dtype, use_scaler = torch.bfloat16, False
    if use_amp and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        amp_dtype, use_scaler = torch.float16, True
        print("[sft] bf16 tidak didukung; memakai fp16 autocast + GradScaler.")
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    steps_per_epoch = math.ceil(len(train_ex) / args.batch_size / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _warmup_cosine_factor(args.warmup_steps, total_steps, args.min_lr_ratio)
    )
    print(f"[sft] contoh={len(train_ex)} val={len(val_ex)} batch={args.batch_size}x{args.grad_accum} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} lr={args.lr} device={device}")

    save_dir = Path(args.save_dir)
    base = evaluate(model, val_ex, pad_id, device, args.batch_size, args.val_batches, use_amp, amp_dtype)
    print(f"[sft] validation loss sebelum SFT (jawaban saja): {base:.4f} ppl={math.exp(base):.2f}")
    best, step, t0, run_loss, run_n = base, 0, time.time(), 0.0, 0

    def save(name):
        _save_checkpoint(save_dir / name, model, tokenizer, optimizer, 0, step, args.env, cfg,
                         scaler=scaler, scheduler=scheduler, training_state={"sft": True, "val_loss": best})

    done = False
    for epoch in range(10 ** 6):
        batches = make_batches(train_ex, args.batch_size, True, args.seed + epoch)
        for bi in range(0, len(batches), args.grad_accum):
            group = batches[bi:bi + args.grad_accum]
            for b in group:
                batch = collate(train_ex, b, pad_id, device)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    loss, _ = model.forward_batch(batch)
                if not torch.isfinite(loss):
                    print("[sft] loss non-finite; batch dilewati")
                    continue
                run_loss += float(loss.detach()); run_n += 1
                (scaler.scale(loss / len(group)) if use_scaler else loss / len(group)).backward()
            _optimizer_step(model, optimizer, scaler, use_scaler, args.grad_clip)
            scheduler.step()
            step += 1
            if step % args.log_every == 0:
                print(f"epoch={epoch+1} step={step}/{total_steps} loss={run_loss/max(run_n,1):.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} elapsed={time.time()-t0:.0f}s", flush=True)
                run_loss, run_n = 0.0, 0
            if step % args.val_every == 0 or step == total_steps:
                v = evaluate(model, val_ex, pad_id, device, args.batch_size, args.val_batches, use_amp, amp_dtype)
                print(f"[sft] step={step} validation_loss={v:.4f} ppl={math.exp(v):.2f} (sebelum SFT {base:.4f})", flush=True)
                if v < best:
                    best = v
                    save("gpt_sft_best.pt")
                    print(f"[sft] best baru; disimpan {save_dir/'gpt_sft_best.pt'}")
            if step % args.save_every == 0:
                save("gpt_sft_last.pt")
            if step >= total_steps:
                done = True
                break
        if done:
            break
    save("gpt_sft_last.pt")
    print(f"[sft] selesai: step={step} best_val={best:.4f} baseline={base:.4f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
