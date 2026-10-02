#!/usr/bin/env python3
"""Sample 500K lines from corpus, split train/val, and pretokenize."""

import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "corpus.txt"
TRAIN_OUT = PROJECT_ROOT / "data" / "corpus_train.txt"
VAL_OUT = PROJECT_ROOT / "data" / "corpus_val.txt"
DEV_DIR = PROJECT_ROOT / "data" / "dev"

SEED = 42
SAMPLE_SIZE = 500000
TRAIN_RATIO = 0.9

def sample_and_split():
    print(f"=== Sampling {SAMPLE_SIZE:,} lines from corpus (seed={SEED}) ===")
    random.seed(SEED)
    
    with open(CORPUS, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total = len(lines)
    print(f"  Total corpus: {total:,} lines")
    
    # Sample
    if total > SAMPLE_SIZE:
        sampled = random.sample(lines, SAMPLE_SIZE)
    else:
        sampled = lines[:]
    
    random.shuffle(sampled)
    
    split_idx = int(len(sampled) * TRAIN_RATIO)
    train_lines = sampled[:split_idx]
    val_lines = sampled[split_idx:]
    
    with open(TRAIN_OUT, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(VAL_OUT, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"  Sampled: {len(sampled):,}")
    print(f"  Train:   {len(train_lines):,} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:     {len(val_lines):,} ({(1-TRAIN_RATIO)*100:.0f}%)")
    return True

def pretokenize(corpus_path, output_path, label):
    print(f"\n=== Pretokenizing {label} ===")
    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "dataset_pretokenizer.py"),
        "--env", "development",
        "--corpus-path", str(corpus_path),
        "--output-path", str(output_path),
        "--overwrite",
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0

def main():
    if not CORPUS.exists():
        print(f"Error: {CORPUS} not found")
        return 1
    
    # Step 1: Sample and split
    if not sample_and_split():
        return 1
    
    # Step 2: Pretokenize train
    train_bin = DEV_DIR / "tokens_train.bin"
    if not pretokenize(TRAIN_OUT, train_bin, "TRAIN"):
        return 1
    
    # Step 3: Pretokenize val
    val_bin = DEV_DIR / "tokens_val.bin"
    if not pretokenize(VAL_OUT, val_bin, "VAL"):
        return 1
    
    print("\n=== DONE ===")
    print(f"Train tokens: {train_bin}")
    print(f"Val tokens:   {val_bin}")
    
    # Show sizes
    for f in [train_bin, val_bin]:
        if f.exists():
            print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
