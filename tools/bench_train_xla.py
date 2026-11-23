"""
Ukur kecepatan latih di TPU (torch_xla, PJRT) dengan arsitektur dan opsi yang sama seperti tools/bench_train.py.

Mode bawaan `spmd`: SATU proses mengendalikan semua chip. Batch dipecah antar chip (sumbu 'data'); model direplikasi di tiap chip,
kecuali bila --shard-params (percobaan: parameter dipecah antar chip, mirip FSDP, agar model yang tidak muat satu chip bisa dicoba).
Mode `mp` (satu proses per chip, torch_xla.launch) gagal di TPU Kaggle ('Expected 8 worker addresses, got 1') dan hanya disimpan sebagai opsi.

  python tools/bench_train_xla.py --preset tiny --ctx 64 --batch 2 --steps 3           # uji asap dulu (menit, hemat kuota)
  python tools/bench_train_xla.py --preset 540m --ctx 1024 --batch 2 --grad-checkpoint # --batch = PER CHIP

Catatan: kode di sini belum teruji penuh di TPU (tidak ada TPU di komputer pengembang). Bagian murni dan logika pekerja diuji dengan tiruan torch_xla di CPU.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# puncak bf16 per chip (TFLOPs, lembar data Google; dipakai hanya untuk MFU kasar)
PEAK_BF16_PER_CHIP = [("v6", 918.0), ("v5p", 459.0), ("v5", 197.0), ("v4", 275.0), ("v3", 123.0), ("v2", 45.0)]


def peak_tflops_per_chip(accelerator_type):
    """'v5litepod-8' / 'v5e-8' / 'v5e' -> 197; 'v6e-8' -> 918; tidak dikenal -> None."""
    a = (accelerator_type or "").lower()
    if a.startswith("v5p"):
        return 459.0
    for prefix, tf in PEAK_BF16_PER_CHIP:
        if a.startswith(prefix):
            return tf
    return None


def summarize(step_times, tokens_per_step_global, flops_per_token, n_chips, peak_per_chip):
    med = statistics.median(step_times)
    tps = tokens_per_step_global / med
    eff = flops_per_token * tps / 1e12
    return {"detik_per_langkah_median": round(med, 3), "token_per_detik": round(tps), "tflops_efektif_total": round(eff, 1),
            "mfu_persen": round(100 * eff / (peak_per_chip * n_chips), 1) if peak_per_chip else None,
            "jam_untuk_10_miliar_token": round(1e10 / tps / 3600, 1)}


def make_checkpoint_patch(xla_checkpoint):
    """Pembungkus untuk torch.utils.checkpoint.checkpoint agar memakai versi XLA (menerima dan membuang use_reentrant yang tidak dikenalnya)."""
    def patched(function, *args, use_reentrant=None, **kwargs):
        kwargs.pop("preserve_rng_state", None)
        return xla_checkpoint(function, *args)
    return patched


def _setup(args):
    """Kembalikan (xm, device, world, rank, mesh, xs). Mode spmd: satu proses, mesh 1 sumbu 'data' atas semua chip."""
    import torch_xla.core.xla_model as xm
    mesh = xs = None
    if args.mode == "spmd":
        import numpy as np
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
        xr.use_spmd()                                   # harus sebelum tensor XLA pertama dibuat
        world, rank = int(xr.global_runtime_device_count()), 0
        mesh = xs.Mesh(np.arange(world), (world,), ("data",))
    else:
        try:
            import torch_xla.runtime as xr
            world, rank = xr.world_size(), xr.global_ordinal()
        except Exception:
            world, rank = xm.xrt_world_size(), xm.get_ordinal()
    return xm, xm.xla_device(), world, rank, mesh, xs


def _worker(index, args):
    import torch
    import bench_train as BT
    from gptModel import GPTModel

    xm, device, world, rank, mesh, xs = _setup(args)
    spmd = args.mode == "spmd"
    cfg = BT.build_cfg(args.preset, args.ctx, dropout=args.dropout)
    torch.manual_seed(0)
    model, tok = GPTModel.from_config_with_tokenizer(cfg=cfg, load_tokenizer=True)
    n = BT.count_params(model)
    d, _h, _kv, layers = BT.PRESETS[args.preset]
    model.to(device).train()
    if spmd and args.shard_params:                      # percobaan: pecah dimensi pertama tiap parameter antar chip bila habis dibagi
        for p in model.parameters():
            if p.dim() >= 1 and p.shape[0] % world == 0:
                xs.mark_sharding(p, mesh, ("data",) + (None,) * (p.dim() - 1))
    if args.grad_checkpoint:
        import torch.utils.checkpoint as tuc
        from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
        tuc.checkpoint = make_checkpoint_patch(xla_checkpoint)
        model.transformer.gradient_checkpointing = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    vocab = len(tok.token2id)
    global_batch = args.batch * world if spmd else args.batch     # spmd: batch global dipecah antar chip; mp: batch per proses
    times, err = [], None
    try:
        for step in range(args.steps + args.warmup):
            t0 = time.time()
            for _ in range(args.grad_accum):
                ids = torch.randint(4, vocab, (global_batch, args.ctx), device=device)
                doc = torch.zeros_like(ids)
                if spmd:
                    xs.mark_sharding(ids, mesh, ("data", None))
                    xs.mark_sharding(doc, mesh, ("data", None))
                batch = {"input_ids": ids, "labels": ids, "document_ids": doc}
                with torch.autocast("xla", dtype=torch.bfloat16):
                    loss = model.forward_batch(batch)[0] / args.grad_accum
                loss.backward()
            if spmd:
                opt.step()                       # gradien sudah dirata-ratakan oleh partisi SPMD
            else:
                xm.optimizer_step(opt)           # mp: rata-rata gradien antar proses lalu langkah optimizer
            opt.zero_grad(set_to_none=True)
            xm.mark_step()
            xm.wait_device_ops()
            if step >= args.warmup:
                times.append(time.time() - t0)
    except Exception as e:                       # kehabisan memori chip muncul sebagai RuntimeError RESOURCE_EXHAUSTED
        err = f"{type(e).__name__}: {str(e).splitlines()[0][:180]}"
    if rank != 0:
        return
    accel = args.accelerator or os.environ.get("TPU_ACCELERATOR_TYPE", "")
    res = {"preset": args.preset, "parameter_juta": round(n / 1e6, 1), "konteks": args.ctx, "batch_per_chip": args.batch, "grad_accum": args.grad_accum,
           "chip": world, "mode": args.mode, "shard_params": bool(args.shard_params), "grad_checkpoint": bool(args.grad_checkpoint), "perangkat": accel or "TPU"}
    if err or not times:
        res["galat"] = err or "tidak ada langkah terukur"
    else:
        ftok = BT.flops_per_token(n, layers, d, args.ctx)
        res.update(summarize(times, args.batch * args.ctx * args.grad_accum * world, ftok, world, peak_tflops_per_chip(accel)))
        try:
            info = xm.get_memory_info(device)
            res["memori_dipakai_gb"] = round(info.get("bytes_used", info.get("kb_total", 0) * 1024) / 1e9, 2)
        except Exception:
            pass
    print(json.dumps(res, ensure_ascii=False), flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


def parse_args(argv=None):
    import bench_train as BT
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(BT.PRESETS), default="100m")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8, help="batch PER CHIP")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3, help="langkah awal tidak dihitung (kompilasi XLA lambat di langkah pertama)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--mode", choices=["spmd", "mp"], default="spmd")
    ap.add_argument("--shard-params", action="store_true", help="percobaan (hanya spmd): pecah parameter antar chip")
    ap.add_argument("--accelerator", default="", help="mis. v5e atau v6e, untuk MFU bila TPU_ACCELERATOR_TYPE tidak terisi")
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "spmd":
        _worker(0, args)
        return
    import torch_xla
    if hasattr(torch_xla, "launch"):
        torch_xla.launch(_worker, args=(args,))
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_worker, args=(args,))


if __name__ == "__main__":
    main()
