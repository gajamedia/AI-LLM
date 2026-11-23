"""
Ukur efisiensi pelatihan (token/detik, memori puncak, MFU) untuk arsitektur dan opsi tertentu, di perangkat apa pun.
Dipakai untuk pintu tahap B (efisiensi) dan untuk memperkirakan biaya model besar dari angka TERUKUR, bukan tebakan.

  python tools/bench_train.py --preset 100m --device cuda --amp                        # garis dasar (seperti pelatihan v2)
  python tools/bench_train.py --preset 540m --device cuda --amp --grad-checkpoint --compile --fused-adamw
  python tools/bench_train.py --preset tiny --device cpu --steps 3                     # uji asap tanpa GPU

Preset parameter dihitung dengan kode model sendiri (lihat rencana): 100m=100,9 jt; 300m=305 jt; 470m=472 jt; 540m=543 jt; 1b=1.061 jt.
MFU = (FLOP per token x token/detik) / puncak FLOPs perangkat; FLOP per token ~ 6 x parameter + 12 x lapisan x d x konteks (atensi).
"""
import argparse
import contextlib
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# nama: (d_model, kepala, kepala KV, lapisan)
PRESETS = {
    "tiny": (64, 4, 2, 2),
    "100m": (768, 12, 4, 12),
    "300m": (1024, 16, 4, 24),
    "470m": (1280, 20, 5, 24),
    "540m": (1280, 20, 5, 28),
    "650m": (1536, 24, 6, 24),
    "970m": (2048, 32, 8, 20),
    "1b": (2048, 32, 8, 22),
}
# puncak FLOPs presisi rendah per perangkat (TFLOPs, dari lembar data; pendekatan untuk MFU)
PEAK_TFLOPS = {"T4": 65.0, "P100": 18.7, "V100": 125.0, "L4": 121.0, "A10": 125.0, "A100": 312.0, "H100": 989.0, "RTX 3090": 142.0, "RTX 4090": 330.0}


def build_cfg(preset, ctx, vocab_dir="data/prod_v2", dropout=0.1):
    from config import get_config
    d, h, kv, L = PRESETS[preset]
    cfg = get_config("production_100m")
    cfg.EMBEDDING_DIM, cfg.ATTENTION_NUM_HEADS, cfg.ATTENTION_NUM_KV_HEADS, cfg.TRANSFORMER_NUM_LAYERS = d, h, kv, L
    cfg.EMBEDDING_MAX_SEQ_LEN = cfg.ATTENTION_MAX_SEQ_LEN = ctx
    cfg.VOCAB_PATH, cfg.TOKEN2ID_PATH, cfg.ID2TOKEN_PATH = (str(PROJECT_ROOT / vocab_dir / f) for f in ("vocab_prod.json", "token2id_prod.json", "id2token_prod.json"))
    cfg.TRANSFORMER_DROPOUT = cfg.EMBEDDING_DROPOUT = cfg.ATTENTION_DROPOUT = dropout
    return cfg


def count_params(model):
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def flops_per_token(n_params, layers, d, ctx):
    return 6 * n_params + 12 * layers * d * ctx


def retie_weights(model):
    """Beberapa backend (mis. torch_tpu) memisahkan bobot terikat saat model dipindah/diubah dtype: embedding dan kepala keluaran jadi dua matriks
    (parameter bertambah vocab x d; model tidak lagi sama dengan checkpoint). Sambungkan lagi."""
    if getattr(model, "tie_weights", False) and hasattr(model, "lm_head") and hasattr(model, "token_embedding"):
        model.lm_head.weight = model.token_embedding.weight


def sync_device(device, model):
    """Tunggu semua kerja perangkat selesai supaya waktu langkah jujur (perangkat async)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type != "cpu":      # mis. 'tpu': membaca satu nilai parameter ke CPU memaksa langkah optimizer selesai
        next(model.parameters()).detach().flatten()[:1].float().cpu()


def detect_peak(device):
    if device.type != "cuda":
        return None, None
    name = torch.cuda.get_device_name(device)
    for key, tf in PEAK_TFLOPS.items():
        if key.lower() in name.lower():
            return name, tf
    return name, None


class _BatchForward(torch.nn.Module):
    """Agar DDP memicu hook gradien: loop memanggil forward_batch, bukan forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.forward_batch(batch)


def run(args):
    from gptModel import GPTModel
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:                                               # multi-GPU lewat torchrun: satu proses per GPU
        import datetime
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=10))
        args.device = f"cuda:{local_rank}"
    device = torch.device(args.device)
    cfg = build_cfg(args.preset, args.ctx, dropout=args.dropout)
    torch.manual_seed(0)
    model, tok = GPTModel.from_config_with_tokenizer(cfg=cfg, load_tokenizer=True)
    model.to(device).train()
    retie_weights(model)
    n = count_params(model)
    d, _h, _kv, L = PRESETS[args.preset]
    amp_ok = True
    if args.amp and device.type not in ("cuda", "cpu"):        # perangkat lain (mis. 'tpu'): autocast bisa tidak tersedia -> pakai bf16 murni
        try:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                pass
        except Exception:
            amp_ok = False
    pure_bf16 = bool(args.amp and not amp_ok)
    if pure_bf16:
        model.to(torch.bfloat16)
        retie_weights(model)                                    # catatan: untuk mengukur kecepatan; latihan sungguhan perlu bobot induk fp32
    if args.grad_checkpoint:
        model.transformer.gradient_checkpointing = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, **({"fused": True} if args.fused_adamw and device.type == "cuda" else {}))
    amp_dtype = torch.bfloat16
    use_scaler = False
    if args.amp and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        amp_dtype, use_scaler = torch.float16, True
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    fwd = torch.compile(model.forward_batch) if args.compile else model.forward_batch
    ddp = None
    if world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(_BatchForward(model), device_ids=[local_rank], output_device=local_rank)
        fwd = ddp
    vocab = len(tok.token2id)
    g = torch.Generator().manual_seed(1 + local_rank)
    def batch():
        ids = torch.randint(4, vocab, (args.batch, args.ctx), generator=g)
        return {"input_ids": ids.to(device), "labels": ids.to(device), "document_ids": torch.zeros_like(ids).to(device)}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for step in range(args.steps + args.warmup):
        t0 = time.time()
        for micro in range(args.grad_accum):
            with (ddp.no_sync() if ddp is not None and micro + 1 < args.grad_accum else contextlib.nullcontext()):
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp and amp_ok):
                    loss = fwd(batch())[0] / args.grad_accum
                (scaler.scale(loss).backward() if use_scaler else loss.backward())
        if use_scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sync_device(device, model)
        if step >= args.warmup:
            times.append(time.time() - t0)
    tokens_step = args.batch * args.ctx * args.grad_accum
    med = statistics.median(times)
    tps_gpu = tokens_step / med                                 # per GPU
    tps = tps_gpu * world                                       # total semua GPU
    ftok = flops_per_token(n, L, d, args.ctx)
    name, peak = detect_peak(device)
    if peak is None and args.peak_tflops:
        name, peak = name or args.device_name or device.type, args.peak_tflops
    res = {"preset": args.preset, "parameter_juta": round(n / 1e6, 1), "konteks": args.ctx, "batch": args.batch, "grad_accum": args.grad_accum,
           "amp": bool(args.amp), "bf16_murni": pure_bf16, "grad_checkpoint": bool(args.grad_checkpoint), "compile": bool(args.compile), "fused_adamw": bool(args.fused_adamw),
           "perangkat": name or str(device), "detik_per_langkah_median": round(med, 3), "token_per_detik": round(tps),
           "tflops_efektif": round(ftok * tps_gpu / 1e12, 2), "mfu_persen": round(100 * ftok * tps_gpu / (peak * 1e12), 1) if peak else None,
           "jumlah_gpu": world, "token_per_detik_per_gpu": round(tps_gpu),
           "memori_puncak_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else None,
           "jam_untuk_10_miliar_token": round(1e10 / tps / 3600, 1)}
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="100m")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fused-adamw", action="store_true")
    ap.add_argument("--peak-tflops", type=float, default=None, help="puncak TFLOPs perangkat non-CUDA untuk MFU (mis. 197 untuk TPU v5e)")
    ap.add_argument("--device-name", default=None, help="nama perangkat di laporan bila tidak terdeteksi (mis. 'TPU v5e-1')")
    ap.add_argument("--out", default=None, help="tambahkan satu baris JSON ke berkas ini")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        res = run(args)
    except torch.cuda.OutOfMemoryError:
        res = {"preset": args.preset, "batch": args.batch, "konteks": args.ctx, "grad_checkpoint": args.grad_checkpoint, "galat": "kehabisan memori GPU (OOM)"}
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:      # jangan biarkan rank lain menunggu selamanya di all-reduce
            if os.environ.get("RANK", "0") == "0":
                print(json.dumps(res, ensure_ascii=False), flush=True)
                if args.out:
                    with open(args.out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res, ensure_ascii=False) + chr(10))
            os._exit(0)
    if os.environ.get("RANK", "0") != "0":                    # multi-GPU: hanya rank 0 melapor
        return res
    print(json.dumps(res, ensure_ascii=False))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    return res


if __name__ == "__main__":
    main()
