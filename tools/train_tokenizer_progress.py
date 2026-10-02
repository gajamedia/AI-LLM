#!/usr/bin/env python3
"""
Train Hybrid Tokenizer with full progress tracking and memory optimization.

Usage:
    python tools/train_tokenizer_progress.py --env development
    python tools/train_tokenizer_progress.py --env production
    python tools/train_tokenizer_progress.py --env development --force
"""

import sys
import argparse
from pathlib import Path

# 1. Setup Path agar bisa load modul dari folder /apps
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import get_config
from tokenizer2 import build_tokenizer_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid Tokenizer with Alignment and Progress Tracking"
    )
    parser.add_argument(
        "--env",
        default="development",
        choices=["development", "production", "testing"],
        help="Environment config yang digunakan (jalur file & hyperparams)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite vocab/token files jika sudah ada (tanpa peringatan)."
    )
    parser.add_argument(
        "--corpus-path",
        default=None,
        help="Corpus used to train tokenizer. Default: environment corpus path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for tokenizer artifacts. Defaults to the selected environment's artifact directory.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Target vocabulary size. Defaults to the selected environment configuration.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        default=True,
        help="Tampilkan progress bar tqdm selama training (default: True)."
    )
    parser.add_argument(
        "--no-show-progress",
        action="store_true",
        help="Nonaktifkan progress bar."
    )
    parser.add_argument(
        "--max-vocab-samples",
        type=int,
        default=None,
        help="Maks jumlah baris corpus untuk initial vocab building. Default dari config."
    )
    parser.add_argument(
        "--max-counter-size",
        type=int,
        default=None,
        help="Batas maks entry Counter sebelum pruning. Default dari config."
    )
    parser.add_argument(
        "--pair-subsample-ratio",
        type=float,
        default=None,
        help="Fraction corpus untuk pair scoring (0.0-1.0). Default dari config."
    )
    parser.add_argument(
        "--max-merges",
        type=int,
        default=None,
        help="Batas maksimum merge rounds. Default dari config (cfg.MAX_MERGES)."
    )
    args = parser.parse_args()

    # Resolve show_progress
    if args.no_show_progress:
        show_progress = False
    else:
        show_progress = args.show_progress

    # 2. Load Config Terpusat
    cfg = get_config(args.env)
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.BASE_DIR)
    output_paths = {
        "vocab": output_dir / Path(cfg.VOCAB_PATH).name,
        "token2id": output_dir / Path(cfg.TOKEN2ID_PATH).name,
        "id2token": output_dir / Path(cfg.ID2TOKEN_PATH).name,
    }
    vocab_size = int(args.vocab_size if args.vocab_size is not None else cfg.VOCAB_SIZE)
    if vocab_size < 4:
        parser.error("--vocab-size must be at least 4 for special tokens.")

    # 3. Resolve memory optimization params
    max_vocab_samples = args.max_vocab_samples or getattr(cfg, "MAX_VOCAB_SAMPLES", None)
    max_counter_size = args.max_counter_size or getattr(cfg, "MAX_COUNTER_SIZE", None)
    pair_subsample_ratio = args.pair_subsample_ratio if args.pair_subsample_ratio is not None else getattr(cfg, "PAIR_SUBSAMPLE_RATIO", None)
    max_merges = int(args.max_merges) if args.max_merges is not None else int(getattr(cfg, "MAX_MERGES", 0) or 0)

    # 4. Cek file output sudah ada
    files_to_check = list(output_paths.values())
    existing = [f for f in files_to_check if f.exists()]
    if existing and not args.force:
        print("[!] File output berikut sudah ada. Gunakan --force untuk overwrite:")
        for f in existing:
            print(f"    - {f}")
        sys.exit(1)

    # 5. Path Alignment
    corpus_path = Path(args.corpus_path) if args.corpus_path else cfg.CORPUS_FILE
    if not corpus_path.exists():
        print(f"[!] Error: Corpus tidak ditemukan di {corpus_path}")
        sys.exit(1)

    # 6. Tokenizer Initialization
    tokenizer = build_tokenizer_from_config(cfg)

    print(f"[*] Mode: {args.env}")
    print(f"[*] Target Vocab: {vocab_size}")
    print(f"[*] Hybrid Params: Alpha={cfg.ALPHA}, Beta={cfg.BETA}, Gamma={cfg.GAMMA}")
    print(f"[*] Memory Opt: max_vocab_samples={max_vocab_samples}, max_counter_size={max_counter_size}, pair_subsample_ratio={pair_subsample_ratio}, max_merges={max_merges}")
    print(f"[*] Corpus: {corpus_path}")
    print(f"[*] Output dir: {output_dir}")
    print()

    # 7. Pastikan output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # 8. Train tokenizer
    print(f"[*] Melatih tokenizer menggunakan: {corpus_path}...")
    tokenizer.train(
        str(corpus_path),
        vocab_size=vocab_size,
        alpha=cfg.ALPHA,
        beta=cfg.BETA,
        gamma=cfg.GAMMA,
        min_pair_freq=cfg.MIN_PAIR_FREQ,
        max_merges=max_merges,
        prune_every=cfg.PRUNE_EVERY,
        show_progress=show_progress,
        progress_desc="Tokenizer train",
        max_vocab_samples=max_vocab_samples,
        max_counter_size=max_counter_size,
        pair_subsample_ratio=pair_subsample_ratio,
    )

    print()
    print(f"[*] Vocab akhir: {len(tokenizer.vocab)} tokens (target: {vocab_size})")

    # 9. Validasi: cek apakah target vocab tercapai
    vocab_pct = (len(tokenizer.vocab) / vocab_size) * 100
    if vocab_pct < 90:
        print(f"[!] Warning: Vocab hanya {vocab_pct:.1f}% dari target. "
              "Pertimbangkan menambah data corpus atau menurunkan target vocabulary.")

    # 10. Test encode/decode sederhana
    test_text = "test tokenizer encoding"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded["input_ids"])
    print(f"[*] Encode test: '{test_text}' -> {len(encoded['input_ids'])} tokens")
    print(f"[*] Decode test: '{decoded}'")

    # 11. Save Alignment
    tokenizer.save_all(
        str(output_paths["vocab"]),
        str(output_paths["token2id"]),
        str(output_paths["id2token"])
    )

    print(f"\n[OK] Tokenizer berhasil disimpan di: {output_dir}")
    print(f"    - Vocab: {output_paths['vocab']}")
    print(f"    - Token2ID: {output_paths['token2id']}")
    print(f"    - ID2Token: {output_paths['id2token']}")


if __name__ == "__main__":
    main()
