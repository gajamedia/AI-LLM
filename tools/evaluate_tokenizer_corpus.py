#!/usr/bin/env python3
"""Evaluate a saved project tokenizer on sampled source-aware JSONL splits."""

import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import get_config
from tokenizer2 import build_tokenizer_from_config


def sample_jsonl(path: Path, sample_size: int, seed: int):
    rng = random.Random(seed)
    reservoir = []
    eligible = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            eligible += 1
            item = (line_number, text)
            if len(reservoir) < sample_size:
                reservoir.append(item)
            else:
                index = rng.randrange(eligible)
                if index < sample_size:
                    reservoir[index] = item
    reservoir.sort(key=lambda item: item[0])
    return eligible, reservoir


def evaluate_split(tokenizer, path: Path, sample_size: int, seed: int):
    eligible, sample = sample_jsonl(path, sample_size, seed)
    totals = {
        "sampled_documents": len(sample),
        "characters": 0,
        "tokens": 0,
        "unknown_tokens": 0,
        "documents_roundtrip_exact_after_whitespace_normalization": 0,
    }
    examples = []
    for line_number, text in sample:
        normalized = re.sub(r"\s+", " ", text).strip()
        tokens = tokenizer.encode_tokens(text)
        unknown = sum(token not in tokenizer.vocab for token in tokens)
        decoded = tokenizer.decode_tokens(tokens)
        totals["characters"] += len(normalized)
        totals["tokens"] += len(tokens)
        totals["unknown_tokens"] += unknown
        exact = decoded == normalized
        totals["documents_roundtrip_exact_after_whitespace_normalization"] += int(exact)
        if not exact and len(examples) < 5:
            examples.append({
                "line": line_number,
                "source_preview": normalized[:180],
                "decoded_preview": decoded[:180],
            })

    docs = totals["sampled_documents"]
    tokens = totals["tokens"]
    return {
        "input": str(path),
        "eligible_documents": eligible,
        **totals,
        "tokens_per_character": round(tokens / max(1, totals["characters"]), 6),
        "unknown_token_rate": round(totals["unknown_tokens"] / max(1, tokens), 8),
        "roundtrip_exact_rate": round(
            totals["documents_roundtrip_exact_after_whitespace_normalization"] / max(1, docs), 6
        ),
        "roundtrip_failure_examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", default="development",
                        choices=["development", "production", "testing"],
                        help="Environment config yang menentukan nama artefak & parameter tokenizer.")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = get_config(args.env)
    tokenizer = build_tokenizer_from_config(cfg)
    tokenizer.load_all(
        str(args.tokenizer_dir / Path(cfg.VOCAB_PATH).name),
        str(args.tokenizer_dir / Path(cfg.TOKEN2ID_PATH).name),
        str(args.tokenizer_dir / Path(cfg.ID2TOKEN_PATH).name),
    )

    report = {
        "tokenizer_dir": str(args.tokenizer_dir),
        "env": args.env,
        "vocab_size": len(tokenizer.vocab),
        "sample_size_per_split": args.sample_size,
        "seed": args.seed,
        "normalization": "collapse whitespace and strip for round-trip comparison",
        "splits": {
            "train": evaluate_split(tokenizer, args.train, args.sample_size, args.seed),
            "validation": evaluate_split(tokenizer, args.validation, args.sample_size, args.seed + 1),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
