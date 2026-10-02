#!/usr/bin/env python3
"""Split raw text or source-aware JSONL, train tokenizer on train only, tokenize both."""

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus.txt"
TRAIN_OUT = PROJECT_ROOT / "data" / "corpus_train.txt"
VAL_OUT = PROJECT_ROOT / "data" / "corpus_val.txt"
DEV_DIR = PROJECT_ROOT / "data" / "dev"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_corpus(
    corpus: Path,
    seed: int = 42,
    train_ratio: float = 0.9,
    train_out: Path = TRAIN_OUT,
    val_out: Path = VAL_OUT,
):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    with open(corpus, "r", encoding="utf-8") as f:
        first_line = next((line for line in f if line.strip()), "")
    is_jsonl = corpus.suffix.lower() == ".jsonl" or first_line.lstrip().startswith("{")

    groups = OrderedDict()
    seen_text = set()
    source_row_by_text = {}
    duplicate_count = 0
    missing_source_count = 0
    with open(corpus, "r", encoding="utf-8") as f:
        for row_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            if is_jsonl:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {corpus}:{row_number}: {exc}") from exc
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    raise ValueError(f"JSONL row {row_number} must contain a string 'text' field.")
                text = row["text"].strip()
                source_id = str(row.get("source_id") or "").strip()
                if not source_id:
                    missing_source_count += 1
                    source_id = f"__unattributed_line_{row_number}"
                row = dict(row)
                row["source_id"] = source_id
            else:
                text = line.strip()
                source_id = f"__line_{row_number}"

            normalized = re.sub(r"\s+", " ", text).casefold()
            if not normalized or normalized in seen_text:
                duplicate_count += bool(normalized)
                continue
            seen_text.add(normalized)
            groups.setdefault(source_id, []).append(text)
            if is_jsonl:
                source_row_by_text[normalized] = row

    if missing_source_count:
        print(
            f"[warn] {missing_source_count:,} JSONL rows have no source_id; "
            "those rows will be split independently."
        )
    if len(groups) < 2:
        raise ValueError("At least two distinct source groups are required for train/validation split.")

    # Greedy source-level assignment balances sentence counts without splitting
    # any source group across train and validation.
    rng = random.Random(seed)
    grouped_rows = list(groups.items())
    rng.shuffle(grouped_rows)
    grouped_rows.sort(key=lambda item: len(item[1]), reverse=True)
    train_groups, val_groups = [], []
    train_count = val_count = 0
    for group in grouped_rows:
        if train_count / train_ratio <= val_count / (1.0 - train_ratio):
            train_groups.append(group)
            train_count += len(group[1])
        else:
            val_groups.append(group)
            val_count += len(group[1])

    if not val_groups:
        val_groups.append(train_groups.pop())
        train_count -= len(val_groups[-1][1])
        val_count += len(val_groups[-1][1])

    train_lines = [text for _, lines in train_groups for text in lines]
    val_lines = [text for _, lines in val_groups for text in lines]
    train_out.parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, "w", encoding="utf-8") as f:
        f.writelines(text + "\n" for text in train_lines)
    with open(val_out, "w", encoding="utf-8") as f:
        f.writelines(text + "\n" for text in val_lines)

    source_jsonl_paths = {}
    if is_jsonl:
        for label, group_list, path in (
            ("train", train_groups, train_out),
            ("validation", val_groups, val_out),
        ):
            source_path = path.with_suffix(".jsonl")
            with open(source_path, "w", encoding="utf-8") as f:
                for source_id, lines in group_list:
                    for text in lines:
                        normalized = re.sub(r"\s+", " ", text).casefold()
                        row = dict(source_row_by_text[normalized])
                        row["source_id"] = source_id
                        json.dump(row, f, ensure_ascii=False)
                        f.write("\n")
            source_jsonl_paths[label] = source_path

    source_hash = _sha256(corpus)
    manifest_common = {
        "input_path": str(corpus),
        "input_sha256": source_hash,
        "input_format": "jsonl" if is_jsonl else "text",
        "split_seed": int(seed),
        "train_ratio_target": float(train_ratio),
        "unique_text_rows": len(seen_text),
        "exact_duplicates_removed": int(duplicate_count),
        "source_groups": len(groups),
        "source_grouped_split": bool(is_jsonl and missing_source_count == 0),
    }
    for label, path, group_list, row_count in (
        ("train", train_out, train_groups, train_count),
        ("validation", val_out, val_groups, val_count),
    ):
        manifest = {
            **manifest_common,
            "split": label,
            "rows": row_count,
            "groups": len(group_list),
            "source_ids": sorted(source_id for source_id, _ in group_list) if is_jsonl else None,
            "source_jsonl_path": str(source_jsonl_paths[label]) if label in source_jsonl_paths else None,
            "source_jsonl_sha256": _sha256(source_jsonl_paths[label]) if label in source_jsonl_paths else None,
            "output_sha256": _sha256(path),
        }
        with open(path.with_suffix(path.suffix + ".manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Input format: {'JSONL with source IDs' if is_jsonl else 'plain text (line-level split)'}")
    print(f"Unique rows: {len(seen_text):,}; exact duplicates removed: {duplicate_count:,}")
    print(f"Source groups: train={len(train_groups):,}, validation={len(val_groups):,}")
    print(f"Rows: train={train_count:,}, validation={val_count:,}")
    print(f"Input SHA-256: {source_hash}")
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
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode == 0


def train_tokenizer(corpus_path):
    print("\n=== Training tokenizer on TRAIN split only ===")
    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "train_tokenizer_progress.py"),
        "--env", "development",
        "--corpus-path", str(corpus_path),
        "--force",
        "--no-show-progress",
    ]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CORPUS, help="Plain text or source-aware JSONL corpus.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1
    if not split_corpus(args.input, seed=args.seed, train_ratio=args.train_ratio):
        return 1

    if not train_tokenizer(TRAIN_OUT):
        return 1
    train_bin = DEV_DIR / "tokens_train.bin"
    if not pretokenize(TRAIN_OUT, train_bin, "TRAIN"):
        return 1
    val_bin = DEV_DIR / "tokens_val.bin"
    if not pretokenize(VAL_OUT, val_bin, "VALIDATION"):
        return 1

    print("\n=== DONE ===")
    print(f"Train tokens: {train_bin}")
    print(f"Validation tokens: {val_bin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
