import argparse
import builtins
import contextlib
import datetime
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import get_config
from checkpoint_meta import build_checkpoint_metadata, validate_checkpoint_metadata
from dataset2 import create_dataloader, parse_token_spec
from gptModel import GPTModel


_PROCESS_START = time.time()   # untuk --time-limit-s (hitungan dimulai sejak proses jalan, termasuk memuat data)


def _dist_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist_on() else 0


def _world() -> int:
    return dist.get_world_size() if _dist_on() else 1


def _is_main() -> bool:
    return _rank() == 0


class _BatchForward(torch.nn.Module):
    """Pembungkus agar DDP memicu hook gradiennya: DDP hanya aktif lewat forward(), sedangkan loop latih
    memanggil model.forward_batch(batch). Parameter dipakai bersama dengan model asli."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.forward_batch(batch)


def _sync_any(flag: bool, device: torch.device) -> bool:
    """True di SEMUA rank bila salah satu rank True (mencegah rank berhenti sendiri lalu macet di all-reduce)."""
    if not _dist_on():
        return bool(flag)
    t = torch.tensor([1 if flag else 0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def _resolve_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(req)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=(device.type == "cuda"))
        else:
            moved[k] = v
    return moved


def _save_checkpoint(
    path: Path,
    model: GPTModel,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    env: str,
    cfg,
    scaler=None,
    scheduler=None,
    training_state=None,
) -> None:
    if not _is_main():
        return   # multi-GPU: hanya rank 0 menulis checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_rng = np.random.get_state()
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": int(epoch),
            "step": int(step),
            "env": env,
            "metadata": build_checkpoint_metadata(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                env=env,
            ),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "training_state": training_state or {},
            "rng_state": {
                "python": random.getstate(),
                "numpy": (
                    numpy_rng[0],
                    torch.from_numpy(numpy_rng[1].copy()),
                    int(numpy_rng[2]),
                    int(numpy_rng[3]),
                    float(numpy_rng[4]),
                ),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        str(temp_path),
    )
    os.replace(temp_path, path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # lepas blok cadangan hasil serialisasi agar tidak memicu OOM di langkah berikutnya
    try:   # berkas kecil pendamping: memilih checkpoint terbaru tanpa memuat file bergiga-giga
        import json as _json
        path.with_suffix('.meta.json').write_text(_json.dumps({'epoch': int(epoch), 'step': int(step), 'env': env, 'saved_at': time.time()}), encoding='utf-8')
    except OSError:
        pass


def _log_save_event(
    epoch: int,
    total_epochs: int,
    step: int,
    run_step: int,
    batch: str | int,
    loss: float,
    ma_loss: float | None,
    validation_loss: float | None,
    lr: float,
    start_time: float,
    cfg,
) -> None:
    if not _is_main():
        return
    elapsed = time.time() - start_time
    ppl = math.exp(min(loss, 20.0))
    ma_text = f"{ma_loss:.6f}" if ma_loss is not None else "n/a"
    val_text = f"{validation_loss:.6f}" if validation_loss is not None else "n/a"
    
    log_entry = (
        f"epoch={epoch}/{total_epochs} "
        f"step={step} "
        f"run_step={run_step} "
        f"batch={batch} "
        f"loss={loss:.6f} "
        f"ma_loss={ma_text} "
        f"validation_loss={val_text} "
        f"lr={lr:.6e} "
        f"ppl~={ppl:.3f} "
        f"elapsed={elapsed:.1f}s"
    )
    
    # Buat timestamp berdasarkan start_time proses (misal: 20260527_1012)
    ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(start_time))
    
    # Ambil base path dari config dan sisipkan timestamp ke nama filenya
    base_log_path = Path(getattr(cfg, "TRAIN_SAVE_LOG_PATH", "reports/logs/train_save.log"))
    log_path = PROJECT_ROOT / base_log_path.parent / f"{base_log_path.stem}_{ts_str}{base_log_path.suffix}"
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


def _load_resume_checkpoint(
    path: Path,
    model: GPTModel,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    cfg,
    scaler=None,
    scheduler=None,
) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Resume checkpoint payload must be a dictionary.")

    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Resume checkpoint missing 'model_state_dict'.")
    validate_checkpoint_metadata(
        payload=payload,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        strict=False,
    )
    model.load_compatible_state_dict(model_state, strict=True)

    optimizer_state = payload.get("optimizer_state_dict")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)

    scaler_state = payload.get("scaler_state_dict")
    if scaler is not None and scaler.is_enabled() and isinstance(scaler_state, dict):
        scaler.load_state_dict(scaler_state)
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)
    rng_state = payload.get("rng_state")
    if isinstance(rng_state, dict):
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rng_state.get("numpy") is not None:
            numpy_state = rng_state["numpy"]
            np.random.set_state((
                numpy_state[0],
                numpy_state[1].cpu().numpy().astype(np.uint32, copy=False),
                int(numpy_state[2]),
                int(numpy_state[3]),
                float(numpy_state[4]),
            ))
        if rng_state.get("torch") is not None:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    return {
        "epoch": int(payload.get("epoch", 0)),
        "step": int(payload.get("step", 0)),
        "training_state": payload.get("training_state", {}),
    }


def _is_plateau_scheduler(scheduler) -> bool:
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def _warmup_cosine_factor(warmup_steps: int, total_steps: int, min_lr_ratio: float):
    """LambdaLR multiplier: linear warmup lalu cosine turun ke min_lr_ratio."""
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(int(total_steps), warmup_steps + 1)
    min_lr_ratio = min(1.0, max(0.0, float(min_lr_ratio)))

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return factor


def _optimizer_step(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_scaler: bool,
    grad_clip: float,
    grad_scale: float = 1.0,
) -> None:
    if use_scaler:
        scaler.unscale_(optimizer)
    if grad_scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(grad_scale)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if use_scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)


def _evaluate_validation_loss(
    model: GPTModel,
    dataloader,
    device: torch.device,
    max_batches: int,
    use_amp: bool,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> float | None:
    if not _is_main():
        return None   # multi-GPU: validasi hanya di rank 0; rank lain menunggu di all-reduce langkah berikutnya
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            batch = _to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model.forward_batch(batch)
            if not (isinstance(out, tuple) and len(out) >= 2):
                raise RuntimeError("Model must return (loss, logits) during validation.")

            loss = out[0]
            if not torch.is_tensor(loss):
                raise RuntimeError("Validation loss output is not a tensor.")

            loss_value = float(loss.detach().cpu().item())
            if math.isfinite(loss_value):
                labels = batch.get("labels")
                token_count = int((labels != -100).sum().item()) if labels is not None else int(batch["input_ids"].numel())
                total_loss += loss_value * token_count
                total_tokens += token_count

    model.train()
    if total_tokens == 0:
        return None
    return total_loss / float(total_tokens)


def _update_validation_monitor(
    loss: float | None,
    best_loss: float,
    bad_evals: int,
    min_delta: float,
    patience: int,
    early_stop_enabled: bool,
    scheduler,
    scaler,
    sampler,
    optimizer,
    model,
    tokenizer,
    epoch: int,
    step: int,
    env: str,
    cfg,
    save_dir: Path,
    save_best: bool = True,
) -> tuple[float, int, bool]:
    """Drive LR scheduling, best-checkpoint selection, and early stopping from val loss."""
    if loss is None:
        return best_loss, bad_evals, False

    if _is_plateau_scheduler(scheduler):
        scheduler.step(loss)

    improved = loss < best_loss - min_delta
    if improved and not save_best:
        best_loss, bad_evals = loss, 0   # model besar: checkpoint terbaik tidak ditulis (6,5 GB tiap kali)
    elif improved:
        best_loss, bad_evals = loss, 0
        _save_checkpoint(
            path=save_dir / "gpt_best.pt", model=model, tokenizer=tokenizer,
            optimizer=optimizer, epoch=epoch, step=step, env=env, cfg=cfg,
            scaler=scaler, scheduler=scheduler,
            training_state={
                "sampler": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
                "best_validation_loss": float(loss),
                "early_bad_evals": 0 if improved else int(bad_evals + 1),
                "last_validation_loss": float(loss),
            },
        )
        print(f"[validation] new best loss={loss:.6f}; saved {save_dir / 'gpt_best.pt'}")
    else:
        bad_evals += 1

    should_stop = early_stop_enabled and bad_evals >= patience
    if should_stop:
        print(f"[early-stop] validation loss did not improve for {bad_evals} evaluations")
    return best_loss, bad_evals, should_stop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GPTModel using integrated build_training_components() pipeline."
    )
    parser.add_argument("--env", default=None, help="Config environment: development/production/testing.")
    parser.add_argument("--device", default=None, help="auto/cpu/cuda (default from config).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default from config).")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (default from config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default from config).")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay (default from config).")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping norm (default from config). <=0 disables.")
    parser.add_argument("--log-every", type=int, default=None, help="Log every N steps (default from config).")
    parser.add_argument("--save-dir", default=None, help="Checkpoint output directory (default from config).")
    parser.add_argument("--save-every", type=int, default=None, help="Save every N steps (default from config). 0=disabled.")
    parser.add_argument("--loss-window", type=int, default=None, help="Moving-average loss window size (default from config).")
    parser.add_argument("--plateau", action="store_true", help="Enable ReduceLROnPlateau scheduler.")
    parser.add_argument("--no-plateau", action="store_true", help="Disable ReduceLROnPlateau scheduler.")
    parser.add_argument("--plateau-factor", type=float, default=None, help="LR reduction factor for plateau scheduler.")
    parser.add_argument("--plateau-patience", type=int, default=None, help="Plateau patience in scheduler evaluation windows.")
    parser.add_argument("--plateau-threshold", type=float, default=None, help="Plateau threshold for improvement.")
    parser.add_argument("--plateau-min-lr", type=float, default=None, help="Minimum LR for scheduler.")
    parser.add_argument("--plateau-cooldown", type=int, default=None, help="Cooldown windows for scheduler.")
    parser.add_argument("--scheduler", default=None, choices=["cosine", "plateau", "none"],
                        help="LR scheduler (default: config TRAIN_SCHEDULER).")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Linear warmup steps sebelum cosine (default: config TRAIN_WARMUP_STEPS).")
    parser.add_argument("--min-lr", type=float, default=None,
                        help="LR minimum pada akhir cosine (default: config TRAIN_MIN_LR).")
    parser.add_argument("--no-amp", action="store_true", help="Paksa nonaktifkan mixed precision.")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stop on non-improving moving-average loss.")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stop.")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Early stop patience in eval windows.")
    parser.add_argument("--early-stop-min-delta", type=float, default=None, help="Minimum delta to count as improvement.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from (e.g. reports/checkpoints/gpt_last.pt).")
    parser.add_argument("--init-from", default=None,
                        help="Checkpoint untuk warm start: hanya BOBOT model dimuat; optimizer, scheduler, step, "
                             "dan sampler mulai baru. Tidak boleh dipakai bersama --resume.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for model initialization and data order.")
    parser.add_argument("--vocab-path", default=None, help="Override vocab JSON path for isolated experiments.")
    parser.add_argument("--token2id-path", default=None, help="Override token2id JSON path for isolated experiments.")
    parser.add_argument("--id2token-path", default=None, help="Override id2token JSON path for isolated experiments.")
    parser.add_argument("--token-file", default=None, help="Override token file path.")
    parser.add_argument("--val-token-file", default=None, help="Optional validation token file path.")
    parser.add_argument("--val-every", type=int, default=None, help="Evaluate validation loss every N optimizer steps. 0=disabled.")
    parser.add_argument("--val-steps", type=int, default=None, help="Max validation batches per evaluation. <=0 uses full validation loader.")

    # Tambahan Fitur
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Gradient checkpointing per lapisan (hemat memori model besar; waktu naik ~30%%).")
    parser.add_argument("--compile", action="store_true", help="torch.compile pada forward training (lebih cepat di GPU; checkpoint tetap dari model asli).")
    parser.add_argument("--fused-adamw", action="store_true", help="AdamW versi fused (hanya CUDA).")
    parser.add_argument("--time-limit-s", type=float, default=None,
                        help="Berhenti rapi (checkpoint gpt_last.pt ditulis) di batas langkah optimizer pertama setelah waktu proses melewati nilai ini; untuk sesi Kaggle 12 jam.")
    parser.add_argument("--no-best-save", action="store_true", help="Jangan tulis gpt_best.pt saat validasi membaik (hemat disk untuk model besar).")
    
    # Dataloader overrides
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset-type", default=None, help="packed/sliding")
    parser.add_argument("--stride", type=int, default=None, help="Only for sliding dataset.")
    parser.add_argument("--shuffle", action="store_true", help="Force enable shuffle.")
    parser.add_argument("--no-shuffle", action="store_true", help="Force disable shuffle.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Torch intra-op threads untuk training CPU (default: nilai torch, biasanya setengah core).",
    )
    args = parser.parse_args()

    if args.shuffle and args.no_shuffle:
        raise SystemExit("Choose only one of --shuffle or --no-shuffle.")
    if args.pin_memory and args.no_pin_memory:
        raise SystemExit("Choose only one of --pin-memory or --no-pin-memory.")
    if args.persistent_workers and args.no_persistent_workers:
        raise SystemExit("Choose only one of --persistent-workers or --no-persistent-workers.")
    if args.plateau and args.no_plateau:
        raise SystemExit("Choose only one of --plateau or --no-plateau.")
    if args.early_stop and args.no_early_stop:
        raise SystemExit("Choose only one of --early-stop or --no-early-stop.")
    if args.num_threads is not None and int(args.num_threads) < 1:
        raise SystemExit("--num-threads must be >= 1.")
    if args.num_threads is not None:
        torch.set_num_threads(int(args.num_threads))

    # Multi-GPU: jalankan lewat `torchrun --nproc_per_node=2 tools/train_gptV2.py ...` (membaca WORLD_SIZE/RANK/LOCAL_RANK).
    ddp_world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_rank = int(os.environ.get("RANK", "0"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if ddp_world > 1:
        if not torch.cuda.is_available():
            raise SystemExit("Multi-GPU (torchrun) memerlukan CUDA.")
        if ddp_local_rank >= torch.cuda.device_count():
            raise SystemExit(f"LOCAL_RANK={ddp_local_rank} tetapi hanya {torch.cuda.device_count()} GPU terlihat.")
        torch.cuda.set_device(ddp_local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
        if ddp_rank != 0:
            builtins.print = lambda *a, **k: None   # hanya rank 0 mencetak log
        if args.scheduler == "plateau" or args.plateau:
            raise SystemExit("Multi-GPU belum mendukung scheduler plateau; pakai --scheduler cosine.")
        if args.compile:
            raise SystemExit("--compile belum dipakai bersama multi-GPU; hilangkan --compile.")
        print(f"[train_gpt] DDP aktif: {ddp_world} proses (backend nccl).")

    env = args.env or os.getenv("TOKENIZER_ENV", "production")
    cfg = get_config(env)
    if args.vocab_path:
        cfg.VOCAB_PATH = Path(args.vocab_path)
    if args.token2id_path:
        cfg.TOKEN2ID_PATH = Path(args.token2id_path)
    if args.id2token_path:
        cfg.ID2TOKEN_PATH = Path(args.id2token_path)
    train_seed = args.seed if args.seed is not None else getattr(cfg, "DATASET_SEED", None)
    if train_seed is not None:
        random.seed(int(train_seed))
        np.random.seed(int(train_seed))
        torch.manual_seed(int(train_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(train_seed))

    train_device = str(args.device if args.device is not None else getattr(cfg, "TRAIN_DEVICE", "auto"))
    train_epochs = int(args.epochs if args.epochs is not None else getattr(cfg, "TRAIN_EPOCHS", 1))
    train_max_steps = args.max_steps if args.max_steps is not None else getattr(cfg, "TRAIN_MAX_STEPS", None)
    if train_max_steps is not None:
        train_max_steps = int(train_max_steps)
    train_lr = float(args.lr if args.lr is not None else getattr(cfg, "TRAIN_LR", 3e-4))
    train_weight_decay = float(
        args.weight_decay if args.weight_decay is not None else getattr(cfg, "TRAIN_WEIGHT_DECAY", 0.1)
    )
    train_grad_clip = float(
        args.grad_clip if args.grad_clip is not None else getattr(cfg, "TRAIN_GRAD_CLIP", 1.0)
    )
    train_log_every = int(
        args.log_every if args.log_every is not None else getattr(cfg, "TRAIN_LOG_EVERY", 20)
    )
    train_save_every = int(
        args.save_every if args.save_every is not None else getattr(cfg, "TRAIN_SAVE_EVERY", 0)
    )
    train_save_dir = str(
        args.save_dir if args.save_dir is not None else getattr(cfg, "TRAIN_SAVE_DIR", "reports/checkpoints")
    )
    train_loss_window = int(
        args.loss_window if args.loss_window is not None else getattr(cfg, "TRAIN_LOSS_WINDOW", 50)
    )
    val_token_file = args.val_token_file or getattr(cfg, "TRAIN_VAL_TOKEN_FILE", None)
    if val_token_file is None:
        val_token_file = getattr(cfg, "VALIDATION_TOKEN_FILE", None)
    val_every_default = getattr(cfg, "TRAIN_VAL_EVERY", None)
    train_val_every = args.val_every if args.val_every is not None else val_every_default
    if train_val_every is None:
        train_val_every = train_log_every if val_token_file else 0
    train_val_every = int(train_val_every)
    train_val_steps = int(
        args.val_steps if args.val_steps is not None else getattr(cfg, "TRAIN_VAL_STEPS", 50)
    )
    if train_epochs < 1:
        raise SystemExit("--epochs must be >= 1.")
    if train_max_steps is not None and train_max_steps < 1:
        raise SystemExit("--max-steps must be >= 1 when provided.")
    if train_loss_window <= 0:
        raise SystemExit("--loss-window must be > 0.")
    if train_log_every <= 0:
        raise SystemExit("--log-every must be > 0.")
    if train_save_every < 0:
        raise SystemExit("--save-every must be >= 0.")
    if train_val_every < 0:
        raise SystemExit("--val-every must be >= 0.")
    if args.scheduler is not None:
        scheduler_name = str(args.scheduler).lower()
    elif getattr(cfg, "TRAIN_SCHEDULER", None):
        scheduler_name = str(cfg.TRAIN_SCHEDULER).lower()
    else:
        # Jalur legacy: konfigurasi lama hanya punya sakelar plateau.
        scheduler_name = "plateau" if bool(getattr(cfg, "TRAIN_PLATEAU_ENABLED", True)) else "none"
    if args.plateau:
        scheduler_name = "plateau"
    if args.no_plateau:
        scheduler_name = "none"
    if scheduler_name not in ("cosine", "plateau", "none"):
        raise SystemExit(f"Unknown scheduler '{scheduler_name}'. Use cosine/plateau/none.")
    plateau_enabled = scheduler_name == "plateau"
    warmup_steps = int(
        args.warmup_steps if args.warmup_steps is not None else getattr(cfg, "TRAIN_WARMUP_STEPS", 500)
    )
    min_lr_floor = float(
        args.min_lr if args.min_lr is not None else getattr(cfg, "TRAIN_MIN_LR", 1e-5)
    )
    plateau_factor = float(
        args.plateau_factor if args.plateau_factor is not None else getattr(cfg, "TRAIN_PLATEAU_FACTOR", 0.5)
    )
    plateau_patience = int(
        args.plateau_patience if args.plateau_patience is not None else getattr(cfg, "TRAIN_PLATEAU_PATIENCE", 3)
    )
    plateau_threshold = float(
        args.plateau_threshold if args.plateau_threshold is not None else getattr(cfg, "TRAIN_PLATEAU_THRESHOLD", 1e-4)
    )
    plateau_min_lr = float(
        args.plateau_min_lr if args.plateau_min_lr is not None else getattr(cfg, "TRAIN_PLATEAU_MIN_LR", 1e-6)
    )
    plateau_cooldown = int(
        args.plateau_cooldown if args.plateau_cooldown is not None else getattr(cfg, "TRAIN_PLATEAU_COOLDOWN", 0)
    )
    early_stop_enabled = (
        bool(args.early_stop)
        if (args.early_stop or args.no_early_stop)
        else bool(getattr(cfg, "TRAIN_EARLY_STOP_ENABLED", False))
    )
    if args.no_early_stop:
        early_stop_enabled = False
    early_stop_patience = int(
        args.early_stop_patience if args.early_stop_patience is not None else getattr(cfg, "TRAIN_EARLY_STOP_PATIENCE", 8)
    )
    early_stop_min_delta = float(
        args.early_stop_min_delta if args.early_stop_min_delta is not None else getattr(cfg, "TRAIN_EARLY_STOP_MIN_DELTA", 1e-4)
    )
    train_grad_accum = int(
        args.grad_accum if args.grad_accum is not None else getattr(cfg, "TRAIN_GRAD_ACCUM", 1)
    )

    if train_grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1")

    device = torch.device("cuda", ddp_local_rank) if ddp_world > 1 else _resolve_device(train_device)

    dataloader_kwargs = {"resumable": True, "seed": train_seed}
    if ddp_world > 1:
        dataloader_kwargs["rank"] = ddp_rank
        dataloader_kwargs["world_size"] = ddp_world
        if train_seed is None:
            dataloader_kwargs["seed"] = 1234   # seed HARUS sama di semua rank agar permutasi sama dan data tidak tumpang tindih
    if args.batch_size is not None:
        dataloader_kwargs["batch_size"] = int(args.batch_size)
    if args.block_size is not None:
        dataloader_kwargs["block_size"] = int(args.block_size)
    if args.dataset_type is not None:
        dataloader_kwargs["dataset_type"] = str(args.dataset_type)
    if args.stride is not None:
        dataloader_kwargs["stride"] = int(args.stride)
    if args.num_workers is not None:
        dataloader_kwargs["num_workers"] = int(args.num_workers)
    if args.shuffle:
        dataloader_kwargs["shuffle"] = True
    if args.no_shuffle:
        dataloader_kwargs["shuffle"] = False
    if args.pin_memory:
        dataloader_kwargs["pin_memory"] = True
    if args.no_pin_memory:
        dataloader_kwargs["pin_memory"] = False
    if args.persistent_workers:
        dataloader_kwargs["persistent_workers"] = True
    if args.no_persistent_workers:
        dataloader_kwargs["persistent_workers"] = False

    # Windows spawn multiprocessing can fail for some dataset/file setups.
    # Force single-process loading on Windows for robust default behavior.
    if os.name == "nt":
        requested_workers = int(
            dataloader_kwargs.get("num_workers", getattr(cfg, "DATASET_NUM_WORKERS", 0))
        )
        if requested_workers > 0:
            dataloader_kwargs["num_workers"] = 0
            dataloader_kwargs["persistent_workers"] = False
            print(
                f"[train_gpt] Windows detected: override num_workers {requested_workers} -> 0 "
                "to avoid multiprocessing spawn/pickle errors."
            )

    # pin_memory is only useful when using CUDA.
    if device.type != "cuda" and "pin_memory" not in dataloader_kwargs:
        if bool(getattr(cfg, "DATASET_PIN_MEMORY", False)):
            dataloader_kwargs["pin_memory"] = False
            print("[train_gpt] CPU device detected: override pin_memory True -> False.")

    try:
        components = GPTModel.build_training_components(
            cfg=cfg,
            load_tokenizer=True,
            token_file=args.token_file,
            **dataloader_kwargs,
        )
    except FileNotFoundError:
        raise SystemExit(
            "Tokenizer artifacts or token file not found.\n"
            f"Expected vocab: {cfg.VOCAB_PATH}\n"
            f"Expected token2id: {cfg.TOKEN2ID_PATH}\n"
            f"Expected id2token: {cfg.ID2TOKEN_PATH}\n"
            f"Expected token file: {args.token_file or cfg.DATASET_TOKEN_FILE}"
        )

    model: GPTModel = components["model"].to(device)
    tokenizer = components["tokenizer"]
    dataloader = components["dataloader"]
    validation_dataloader = None
    if val_token_file and train_val_every > 0:
        val_token_path = Path(str(val_token_file))
        for _vp, _rng in parse_token_spec(val_token_file):
            if not Path(_vp).exists():
                raise SystemExit(f"Validation token file not found: {_vp}")
        validation_dataloader = create_dataloader(
            token_file=str(val_token_file),
            block_size=int(dataloader_kwargs.get("block_size", getattr(cfg, "DATASET_BLOCK_SIZE", 128))),
            batch_size=int(dataloader_kwargs.get("batch_size", getattr(cfg, "DATASET_BATCH_SIZE", 16))),
            shuffle=False,
            num_workers=int(dataloader_kwargs.get("num_workers", getattr(cfg, "DATASET_NUM_WORKERS", 0))),
            pin_memory=bool(dataloader_kwargs.get("pin_memory", getattr(cfg, "DATASET_PIN_MEMORY", True))),
            persistent_workers=bool(
                dataloader_kwargs.get("persistent_workers", getattr(cfg, "DATASET_PERSISTENT_WORKERS", False))
            ),
            drop_last=False,
            dataset_type=str(dataloader_kwargs.get("dataset_type", getattr(cfg, "DATASET_TYPE", "packed"))),
            stride=int(dataloader_kwargs.get("stride", getattr(cfg, "SLIDING_STRIDE", 1))),
            dtype=getattr(cfg, "DATASET_DTYPE", None),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=int(tokenizer.token2id[tokenizer.pad_token]),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_lr,
        weight_decay=train_weight_decay,
        **({"fused": True} if args.fused_adamw and device.type == "cuda" else {}),
    )
    if args.grad_checkpoint:
        model.transformer.gradient_checkpointing = True
        print("[train_gpt] gradient checkpointing aktif.")
    # forward yang dipakai loop latih; compile hanya membungkus pemanggilan, model asli tetap yang disimpan ke checkpoint
    forward_batch = torch.compile(model.forward_batch) if args.compile else model.forward_batch
    if args.compile:
        print("[train_gpt] torch.compile aktif pada forward training.")
    ddp_module = None
    if ddp_world > 1:
        ddp_module = torch.nn.parallel.DistributedDataParallel(
            _BatchForward(model), device_ids=[ddp_local_rank], output_device=ddp_local_rank,
        )
        forward_batch = ddp_module
    amp_wanted = (bool(args.amp) or bool(getattr(cfg, "TRAIN_AMP_ENABLED", False))) and not args.no_amp
    use_amp = bool(amp_wanted) and device.type in ("cuda", "cpu")
    amp_dtype = torch.bfloat16
    use_scaler = False
    # is_bf16_supported() bernilai True di T4 karena emulasi (lambat); pakai compute capability >= 8.0.
    if use_amp and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        # T4/P100 (Kaggle/Colab gratis): tanpa bf16 native -> fp16 + GradScaler.
        amp_dtype = torch.float16
        use_scaler = True
        print("[train_gpt] bf16 tidak didukung; memakai fp16 autocast + GradScaler.")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    planned_updates = (
        train_max_steps
        if train_max_steps is not None
        else train_epochs * math.ceil(len(dataloader) / train_grad_accum)
    )

    scheduler = None
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            cooldown=plateau_cooldown,
            min_lr=plateau_min_lr,
        )
    elif scheduler_name == "cosine":
        if warmup_steps >= planned_updates:
            print(
                f"[warn] warmup_steps={warmup_steps} >= planned_optimizer_steps={planned_updates}; "
                "cosine tidak punya ruang untuk turun."
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            _warmup_cosine_factor(
                warmup_steps,
                planned_updates,
                min_lr_floor / train_lr if train_lr > 0 else 0.0,
            ),
        )

    save_dir = Path(train_save_dir)
    global_step = 0
    resume_epoch = 0
    run_step = 0
    run_optimizer_steps = 0
    skipped_non_finite = 0
    start_time = time.time()
    last_heartbeat = start_time
    recent_losses = deque(maxlen=train_loss_window)
    moving_avg_loss = None
    last_validation_loss = None
    loss_value = float("nan")
    best_moving_avg = float("inf")
    best_validation_loss = float("inf")
    early_bad_windows = 0
    stop_training = False
    model.train()

    if args.init_from:
        if args.resume:
            raise SystemExit("--init-from dan --resume tidak boleh dipakai bersamaan.")
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise SystemExit(f"--init-from checkpoint not found: {init_path}")
        init_payload = torch.load(str(init_path), map_location="cpu")
        if not isinstance(init_payload, dict) or not isinstance(init_payload.get("model_state_dict"), dict):
            raise SystemExit("--init-from checkpoint missing 'model_state_dict'.")
        validate_checkpoint_metadata(payload=init_payload, model=model, tokenizer=tokenizer, cfg=cfg, strict=False)
        model.load_compatible_state_dict(init_payload["model_state_dict"], strict=True)
        print(f"[train_gpt] warm start: bobot dimuat dari {init_path} "
              f"(checkpoint_epoch={init_payload.get('epoch')}, checkpoint_step={init_payload.get('step')}); "
              "optimizer/scheduler/sampler baru.")
        del init_payload

    if args.resume:
        resume_info = _load_resume_checkpoint(
            path=Path(args.resume),
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            cfg=cfg,
            scaler=scaler,
            scheduler=scheduler,
        )
        saved_training_state = resume_info.get("training_state", {})
        training_sampler = getattr(dataloader, "sampler", None)
        sampler_state = saved_training_state.get("sampler") if isinstance(saved_training_state, dict) else None
        if hasattr(training_sampler, "load_state_dict") and isinstance(sampler_state, dict):
            training_sampler.load_state_dict(sampler_state)
        best_validation_loss = float(saved_training_state.get("best_validation_loss", float("inf")))
        early_bad_windows = int(saved_training_state.get("early_bad_evals", 0))
        last_validation_loss = saved_training_state.get("last_validation_loss")
        recent_history = saved_training_state.get("recent_losses", [])
        if isinstance(recent_history, list):
            recent_losses.extend(float(value) for value in recent_history[-train_loss_window:])
        resume_epoch = int(resume_info["epoch"])
        if isinstance(sampler_state, dict) and sampler_state.get("permutation") is not None:
            if int(sampler_state.get("consumed_position", 0)) < int(sampler_state["permutation"].numel()):
                resume_epoch -= 1
        global_step = int(resume_info["step"])
        print(
            f"[train_gpt] resumed from {args.resume} "
            f"(checkpoint_epoch={resume_epoch}, checkpoint_step={global_step})"
        )
        print(f"[train_gpt] effective_resume_lr={optimizer.param_groups[0]['lr']:.6e}")

    training_sampler = getattr(dataloader, "sampler", None)

    def training_state_snapshot():
        return {
            "sampler": training_sampler.state_dict() if hasattr(training_sampler, "state_dict") else None,
            "best_validation_loss": best_validation_loss,
            "early_bad_evals": early_bad_windows,
            "last_validation_loss": last_validation_loss,
            "recent_losses": list(recent_losses),
            "moving_avg_loss": moving_avg_loss,
        }

    print("=" * 80)
    print("GPT Training")
    print("=" * 80)
    print(f"env={env}")
    print(f"device={device}")
    print(f"vocab_size={model.vocab_size}")
    print(f"d_model={model.d_model}")
    print(f"num_layers={model.transformer.num_layers}")
    print(f"epochs={train_epochs}")
    print(
        f"max_optimizer_steps={train_max_steps}"
        + (" (kumulatif: mencakup global_step hasil resume)" if args.resume else "")
    )
    print(f"lr={train_lr}")
    print(f"log_every={train_log_every}")
    print(f"loss_window={train_loss_window}")
    print(f"validation_enabled={validation_dataloader is not None}")
    if validation_dataloader is not None:
        print(f"validation_every={train_val_every}")
        print(f"validation_steps={train_val_steps if train_val_steps > 0 else 'all'}")
        print(f"validation_token_file={val_token_file}")
    print(f"scheduler={scheduler_name}")
    if scheduler_name == "cosine":
        print(f"warmup_steps={warmup_steps}")
        print(f"min_lr={min_lr_floor:.3e}")
    print(f"plateau_enabled={plateau_enabled}")
    print(f"early_stop_enabled={early_stop_enabled}")
    print(f"resume={args.resume if args.resume else 'none'}")
    print(f"init_from={args.init_from if args.init_from else 'none'}")
    print(f"mixed_precision={use_amp} dtype={amp_dtype}")
    print(f"num_threads={torch.get_num_threads()}")
    print(f"grad_accum={train_grad_accum}")
    print(f"effective_batch={train_grad_accum * dataloader.batch_size * ddp_world}" + (f" (per GPU {dataloader.batch_size}, {ddp_world} GPU)" if ddp_world > 1 else ""))
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    dataset_tokens = int(getattr(dataloader.dataset, "num_training_tokens", 0))
    samples = len(dataloader.dataset)
    average_target_tokens = dataset_tokens / max(1, samples)
    sequence_slots = int(getattr(dataloader.dataset, "block_size", 0))
    token_utilization = average_target_tokens / max(1, sequence_slots)
    planned_tokens = int(
        planned_updates * dataloader.batch_size * train_grad_accum * ddp_world * average_target_tokens
    )
    planned_passes = planned_tokens / max(1, dataset_tokens)
    print(f"trainable_parameters={trainable_params:,}")
    print(f"training_target_tokens={dataset_tokens:,}")
    print(f"training_tokens_per_parameter={dataset_tokens / max(1, trainable_params):.2f}")
    print(f"average_sequence_token_utilization={token_utilization:.1%}")
    print(f"planned_training_tokens~={planned_tokens:,} ({planned_passes:.1f} dataset passes)")
    if dataset_tokens / max(1, trainable_params) < 10:
        print("[warn] Data is small relative to model size; consider more unique data or a smaller model.")
    if token_utilization < 0.5:
        print("[warn] Document-aware batches are heavily padded; consider packing multiple documents per sequence with boundary-aware attention.")
    if planned_passes > 20:
        print("[warn] Training plan repeats the dataset many times; review epochs/max-steps for overfitting and runtime.")
    print("training started... waiting for first batch/optimizer step")

    last_epoch_number = resume_epoch
    for local_epoch in range(train_epochs):
        epoch_number = resume_epoch + local_epoch + 1
        last_epoch_number = epoch_number
        accum_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(dataloader):
            if hasattr(training_sampler, "mark_consumed"):
                training_sampler.mark_consumed(int(batch["input_ids"].shape[0]))
            batch = _to_device(batch, device)

            # DDP: all-reduce gradien hanya pada mikro-langkah terakhir akumulasi (no_sync di langkah sebelumnya).
            if ddp_module is not None and accum_batches + 1 < train_grad_accum:
                sync_ctx = ddp_module.no_sync()
            else:
                sync_ctx = contextlib.nullcontext()
            with sync_ctx:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = forward_batch(batch)
                if not (isinstance(out, tuple) and len(out) >= 2):
                    raise RuntimeError("Model must return (loss, logits) during training.")
                loss = out[0]
                if not torch.is_tensor(loss):
                    raise RuntimeError("Loss output is not a tensor.")

                loss_value = float(loss.detach().cpu().item())
                # semua rank harus sepakat melewati langkah ini; bila hanya satu rank yang melewatinya, rank lain macet di all-reduce
                if _sync_any(not math.isfinite(loss_value), device):
                    skipped_non_finite += 1
                    if ddp_module is not None:
                        # tuntaskan siklus reduksi DDP dengan backward nol (gradien nol) supaya forward berikutnya tidak galat
                        sum(p.sum() for p in model.parameters() if p.requires_grad).mul(0.0).backward()
                    accum_batches = 0
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        f"[warn] non-finite loss detected (loss={loss_value}) at "
                        f"epoch={epoch_number} batch={batch_idx + 1}; accumulated step dropped. "
                        f"total_skipped={skipped_non_finite}"
                    )
                    continue

                loss = loss / train_grad_accum
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            run_step += 1
            accum_batches += 1
            recent_losses.append(loss_value)
            optimizer_stepped = False

            if accum_batches >= train_grad_accum:
                _optimizer_step(model, optimizer, scaler, use_scaler, train_grad_clip)
                global_step += 1
                run_optimizer_steps += 1
                accum_batches = 0
                optimizer_stepped = True
                if scheduler is not None and not _is_plateau_scheduler(scheduler):
                    scheduler.step()

            if optimizer_stepped and len(recent_losses) == train_loss_window:
                moving_avg_loss = sum(recent_losses) / float(train_loss_window)
                if validation_dataloader is None and _is_plateau_scheduler(scheduler) and global_step % train_loss_window == 0:
                    prev_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step(moving_avg_loss)
                    new_lr = optimizer.param_groups[0]["lr"]
                    if new_lr < prev_lr:
                        print(
                            f"[scheduler] lr reduced: {prev_lr:.6e} -> {new_lr:.6e} "
                            f"(ma_loss={moving_avg_loss:.6f})"
                        )

                if validation_dataloader is None and early_stop_enabled and global_step % train_loss_window == 0:
                    if moving_avg_loss < (best_moving_avg - early_stop_min_delta):
                        best_moving_avg = moving_avg_loss
                        early_bad_windows = 0
                    else:
                        early_bad_windows += 1
                        if early_bad_windows >= early_stop_patience:
                            print(
                                f"[early-stop] triggered at step={global_step} "
                                f"(ma_loss={moving_avg_loss:.6f}, best={best_moving_avg:.6f}, "
                                f"bad_windows={early_bad_windows})"
                            )
                            stop_training = True

            if (
                optimizer_stepped
                and validation_dataloader is not None
                and global_step % train_val_every == 0
            ):
                last_validation_loss = _evaluate_validation_loss(
                    model=model,
                    dataloader=validation_dataloader,
                    device=device,
                    max_batches=train_val_steps,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                best_validation_loss, early_bad_windows, val_stop = _update_validation_monitor(
                    loss=last_validation_loss,
                    best_loss=best_validation_loss,
                    bad_evals=early_bad_windows,
                    min_delta=early_stop_min_delta,
                    patience=early_stop_patience,
                    early_stop_enabled=early_stop_enabled,
                    scheduler=scheduler,
                    scaler=scaler,
                    sampler=training_sampler,
                    optimizer=optimizer,
                    model=model,
                    tokenizer=tokenizer,
                    epoch=epoch_number,
                    step=global_step,
                    env=env,
                    cfg=cfg,
                    save_dir=save_dir,
                    save_best=not args.no_best_save,
                )
                stop_training = stop_training or val_stop

            if optimizer_stepped and (run_optimizer_steps == 1 or global_step % train_log_every == 0):
                ppl = math.exp(min(20.0, loss_value))
                elapsed = time.time() - start_time
                current_lr = optimizer.param_groups[0]["lr"]
                ma_text = f"{moving_avg_loss:.6f}" if moving_avg_loss is not None else "n/a"
                val_text = f"{last_validation_loss:.6f}" if last_validation_loss is not None else "n/a"
                print(
                    f"epoch={epoch_number}/{resume_epoch + train_epochs} "
                    f"step={global_step} "
                    f"run_optimizer_steps={run_optimizer_steps} "
                    f"micro_steps={run_step} "
                    f"batch={batch_idx + 1} "
                    f"loss={loss_value:.6f} "
                    f"ma_loss={ma_text} "
                    f"validation_loss={val_text} "
                    f"lr={current_lr:.6e} "
                    f"ppl~={ppl:.3f} "
                    f"elapsed={elapsed:.1f}s"
                )
                last_heartbeat = time.time()

            now = time.time()
            if now - last_heartbeat >= 30.0:
                elapsed = now - start_time
                print(
                    f"[heartbeat] epoch={epoch_number}/{resume_epoch + train_epochs} "
                    f"step={global_step} "
                    f"run_optimizer_steps={run_optimizer_steps} "
                    f"micro_steps={run_step} "
                    f"batch={batch_idx + 1} "
                    f"elapsed={elapsed:.1f}s "
                    f"skipped_non_finite={skipped_non_finite}"
                )
                last_heartbeat = now

            if optimizer_stepped and train_save_every > 0 and global_step % train_save_every == 0:
                ckpt = save_dir / f"gpt_step_{global_step}.pt"
                _save_checkpoint(
                    path=ckpt,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    epoch=epoch_number,
                    step=global_step,
                    env=env,
                    cfg=cfg,
                    scaler=scaler,
                    scheduler=scheduler,
                    training_state=training_state_snapshot(),
                )
                _log_save_event(
                    epoch=epoch_number,
                    total_epochs=resume_epoch + train_epochs,
                    step=global_step,
                    run_step=run_step,
                    batch=batch_idx + 1,
                    loss=loss_value,
                    ma_loss=moving_avg_loss,
                    validation_loss=last_validation_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    start_time=start_time,
                    cfg=cfg,
                )
                print(f"saved checkpoint: {ckpt}")

            if train_max_steps is not None and global_step >= train_max_steps:
                stop_training = True
            if optimizer_stepped and args.time_limit_s and time.time() - _PROCESS_START >= args.time_limit_s and not stop_training:
                stop_training = True
                print(f"[time-limit] {args.time_limit_s:.0f}s tercapai di step={global_step}; berhenti dan menulis gpt_last.pt")
            if optimizer_stepped:
                stop_training = _sync_any(stop_training, device)   # keputusan berhenti (mis. early-stop di rank 0) harus seragam
            if stop_training:
                break

        if hasattr(training_sampler, "finish_epoch") and (
            not stop_training
            or training_sampler.consumed_position >= len(training_sampler) * getattr(training_sampler, "world_size", 1)
        ):
            training_sampler.finish_epoch()

        if accum_batches > 0 and not stop_training:
            if ddp_module is not None:
                # sisa akumulasi dijalankan di bawah no_sync: rata-ratakan gradien antar rank secara manual
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad.div_(ddp_world)
            _optimizer_step(
                model, optimizer, scaler, use_scaler, train_grad_clip,
                grad_scale=float(train_grad_accum) / float(accum_batches),
            )
            global_step += 1
            run_optimizer_steps += 1
            if scheduler is not None and not _is_plateau_scheduler(scheduler):
                scheduler.step()
            print(
                f"[train_gpt] flushed {accum_batches} accumulated batch(es) at "
                f"epoch={epoch_number}; step={global_step}"
            )

            if len(recent_losses) == train_loss_window:
                moving_avg_loss = sum(recent_losses) / float(train_loss_window)
                if validation_dataloader is None and _is_plateau_scheduler(scheduler) and global_step % train_loss_window == 0:
                    prev_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step(moving_avg_loss)
                    new_lr = optimizer.param_groups[0]["lr"]
                    if new_lr < prev_lr:
                        print(
                            f"[scheduler] lr reduced: {prev_lr:.6e} -> {new_lr:.6e} "
                            f"(ma_loss={moving_avg_loss:.6f})"
                        )

                if validation_dataloader is None and early_stop_enabled and global_step % train_loss_window == 0:
                    if moving_avg_loss < (best_moving_avg - early_stop_min_delta):
                        best_moving_avg = moving_avg_loss
                        early_bad_windows = 0
                    else:
                        early_bad_windows += 1
                        if early_bad_windows >= early_stop_patience:
                            print(
                                f"[early-stop] triggered at step={global_step} "
                                f"(ma_loss={moving_avg_loss:.6f}, best={best_moving_avg:.6f}, "
                                f"bad_windows={early_bad_windows})"
                            )
                            stop_training = True

            if (
                validation_dataloader is not None
                and global_step % train_val_every == 0
            ):
                last_validation_loss = _evaluate_validation_loss(
                    model=model,
                    dataloader=validation_dataloader,
                    device=device,
                    max_batches=train_val_steps,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                best_validation_loss, early_bad_windows, val_stop = _update_validation_monitor(
                    loss=last_validation_loss,
                    best_loss=best_validation_loss,
                    bad_evals=early_bad_windows,
                    min_delta=early_stop_min_delta,
                    patience=early_stop_patience,
                    early_stop_enabled=early_stop_enabled,
                    scheduler=scheduler,
                    scaler=scaler,
                    sampler=training_sampler,
                    optimizer=optimizer,
                    model=model,
                    tokenizer=tokenizer,
                    epoch=epoch_number,
                    step=global_step,
                    env=env,
                    cfg=cfg,
                    save_dir=save_dir,
                    save_best=not args.no_best_save,
                )
                stop_training = stop_training or val_stop

            if run_optimizer_steps == 1 or global_step % train_log_every == 0:
                ppl = math.exp(min(20.0, loss_value))
                elapsed = time.time() - start_time
                current_lr = optimizer.param_groups[0]["lr"]
                ma_text = f"{moving_avg_loss:.6f}" if moving_avg_loss is not None else "n/a"
                val_text = f"{last_validation_loss:.6f}" if last_validation_loss is not None else "n/a"
                print(
                    f"epoch={epoch_number}/{resume_epoch + train_epochs} "
                    f"step={global_step} "
                    f"run_optimizer_steps={run_optimizer_steps} "
                    f"micro_steps={run_step} "
                    f"batch=end "
                    f"loss={loss_value:.6f} "
                    f"ma_loss={ma_text} "
                    f"validation_loss={val_text} "
                    f"lr={current_lr:.6e} "
                    f"ppl~={ppl:.3f} "
                    f"elapsed={elapsed:.1f}s"
                )
                last_heartbeat = time.time()

            if train_save_every > 0 and global_step % train_save_every == 0:
                ckpt = save_dir / f"gpt_step_{global_step}.pt"
                _save_checkpoint(
                    path=ckpt,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    epoch=epoch_number,
                    step=global_step,
                    env=env,
                    cfg=cfg,
                    scaler=scaler,
                    scheduler=scheduler,
                    training_state=training_state_snapshot(),
                )
                _log_save_event(
                    epoch=epoch_number,
                    total_epochs=resume_epoch + train_epochs,
                    step=global_step,
                    run_step=run_step,
                    batch="end",
                    loss=loss_value,
                    ma_loss=moving_avg_loss,
                    validation_loss=last_validation_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    start_time=start_time,
                    cfg=cfg,
                )
                print(f"saved checkpoint: {ckpt}")

        if train_max_steps is not None and global_step >= train_max_steps:
            break
        if stop_training:
            break

    final_ckpt = save_dir / "gpt_last.pt"
    _save_checkpoint(
        path=final_ckpt,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        epoch=last_epoch_number,
        step=global_step,
        env=env,
        cfg=cfg,
        scaler=scaler,
        scheduler=scheduler,
        training_state=training_state_snapshot(),
    )
    _log_save_event(
        epoch=last_epoch_number,
        total_epochs=resume_epoch + train_epochs,
        step=global_step,
        run_step=run_step,
        batch="end",
        loss=loss_value,
        ma_loss=moving_avg_loss,
        validation_loss=last_validation_loss,
        lr=optimizer.param_groups[0]["lr"],
        start_time=start_time,
        cfg=cfg,
    )
    print(f"training complete: micro_steps={run_step}, optimizer_steps={run_optimizer_steps}, global_steps={global_step}")
    print(f"skipped_non_finite={skipped_non_finite}")
    print(f"final checkpoint: {final_ckpt}")
    if _dist_on():
        dist.barrier()   # tunggu rank 0 selesai menulis checkpoint
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
