#!/usr/bin/env python3
"""Deterministic streaming train/validation split with source-group isolation."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-every", type=int, default=10,
                        help="Assign 1 in N stable source hash buckets to validation")
    parser.add_argument("--emit-text", action="store_true",
                        help="Also write train.txt/validation.txt (plain text, one text per line) "
                             "for tools that read .txt instead of .jsonl")
    parser.add_argument("--tokenizer-sample", type=int, default=0,
                        help="Write a uniform random sample of N train rows to tokenizer_sample.txt "
                             "(plain text, shuffled) for tokenizer training")
    parser.add_argument("--tokenizer-sample-seed", type=int, default=17)
    args = parser.parse_args()
    if not 2 <= args.validation_every <= 100:
        parser.error("--validation-every must be between 2 and 100")
    if args.tokenizer_sample < 0:
        parser.error("--tokenizer-sample must be >= 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    counts = {"train": Counter(), "validation": Counter()}
    sources = {"train": set(), "validation": set()}
    chars = Counter()
    train_txt = args.output_dir / "train.txt"
    validation_txt = args.output_dir / "validation.txt"
    sample_path = args.output_dir / "tokenizer_sample.txt"
    sample_n = args.tokenizer_sample
    sample_rng = random.Random(args.tokenizer_sample_seed)
    sample_seen = 0
    sample_reservoir: list = []

    text_train = train_txt.open("w", encoding="utf-8", newline="\n") if args.emit_text else None
    text_validation = validation_txt.open("w", encoding="utf-8", newline="\n") if args.emit_text else None
    try:
        with args.input.open("r", encoding="utf-8") as source, \
                train_path.open("w", encoding="utf-8", newline="\n") as train, \
                validation_path.open("w", encoding="utf-8", newline="\n") as validation:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {args.input}:{line_number}") from exc
                source_id = str(row.get("source_id") or "").strip() if isinstance(row, dict) else ""
                if not source_id or not isinstance(row.get("text"), str):
                    raise ValueError(f"Missing source_id or string text at {args.input}:{line_number}")
                bucket = int.from_bytes(hashlib.sha256(source_id.encode("utf-8")).digest()[:8], "big")
                split = "validation" if bucket % args.validation_every == 0 else "train"
                destination = validation if split == "validation" else train
                destination.write(line.rstrip("\r\n") + "\n")
                counts[split]["rows"] += 1
                counts[split]["characters"] += len(row["text"])
                sources[split].add(source_id)
                if text_train is not None or text_validation is not None:
                    target_txt = text_validation if split == "validation" else text_train
                    if target_txt is not None:
                        target_txt.write(row["text"] + "\n")
                if sample_n and split == "train":
                    sample_seen += 1
                    if len(sample_reservoir) < sample_n:
                        sample_reservoir.append(row["text"])
                    else:
                        slot = sample_rng.randrange(sample_seen)
                        if slot < sample_n:
                            sample_reservoir[slot] = row["text"]
    finally:
        if text_train is not None:
            text_train.close()
        if text_validation is not None:
            text_validation.close()

    if sample_n:
        sample_rng.shuffle(sample_reservoir)
        with sample_path.open("w", encoding="utf-8", newline="\n") as sample_out:
            for text in sample_reservoir:
                sample_out.write(text + "\n")

    if sources["train"] & sources["validation"]:
        raise AssertionError("Source leakage detected between splits")
    manifest = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "validation_bucket_modulus": args.validation_every,
        "split_method": "stable SHA-256 source_id buckets; all records for one source remain together",
        "source_overlap": 0,
        "emit_text": bool(args.emit_text),
        "tokenizer_sample": (
            {"path": str(sample_path), "rows": len(sample_reservoir),
             "seed": args.tokenizer_sample_seed, "sampled_from_train_rows": sample_seen}
            if sample_n else None
        ),
        "train": {"path": str(train_path), "rows": counts["train"]["rows"],
                  "characters": counts["train"]["characters"],
                  "sources": len(sources["train"]), "sha256": sha256(train_path)},
        "validation": {"path": str(validation_path), "rows": counts["validation"]["rows"],
                       "characters": counts["validation"]["characters"],
                       "sources": len(sources["validation"]), "sha256": sha256(validation_path)},
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
