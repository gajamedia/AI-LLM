#!/usr/bin/env python3
"""Report size, shape, duplication and token budget for plain-text or source-aware JSONL corpora.

Streaming and bounded memory: duplication and source tracking use 64-bit digests
instead of raw strings, so a 100M-token corpus does not need gigabytes of RAM.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

DEFAULT_CHAR_BUCKETS = (0, 40, 80, 120, 160, 240, 320, 480, 640, 1000, 2000)
REQUIRED_FIELDS = ("text", "source_id", "source_title", "source_url")


def detect_jsonl(path: Path) -> bool:
    if path.suffix.lower() == ".jsonl":
        return True
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                return line.lstrip().startswith("{")
    return False


def digest64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def bucket_index(length: int, buckets) -> int:
    for index, upper in enumerate(buckets):
        if length <= upper:
            return index
    return len(buckets)


def iter_documents(path: Path, is_jsonl: bool):
    """Yield (text, source_id) pairs. source_id is '' when unavailable."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if not is_jsonl:
                yield line.strip(), f"__line_{line_number}", line_number, None
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            text = row.get("text")
            if not isinstance(text, str):
                yield "", str(row.get("source_id") or ""), line_number, row
                continue
            yield text, str(row.get("source_id") or ""), line_number, row


def analyze(path: Path, skip_dedup: bool) -> dict:
    is_jsonl = detect_jsonl(path)
    buckets = list(DEFAULT_CHAR_BUCKETS)
    histogram = [0] * (len(buckets) + 1)

    rows = rows_with_text = empty_text = 0
    total_chars = whitespace_chars = 0
    min_chars = None
    max_chars = 0
    missing_field_counts = {field: 0 for field in REQUIRED_FIELDS}
    seen_text = set() if not skip_dedup else None
    seen_sources = set()
    duplicate_texts = 0

    for text, source_id, _line_number, row in iter_documents(path, is_jsonl):
        rows += 1
        if row is not None:
            for field in REQUIRED_FIELDS:
                value = row.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing_field_counts[field] += 1
        normalized = re.sub(r"\s+", " ", text).strip() if text else ""
        if not normalized:
            empty_text += 1
            continue
        rows_with_text += 1
        length = len(normalized)
        total_chars += length
        whitespace_chars += len(text) - len("".join(text.split()))
        max_chars = max(max_chars, length)
        min_chars = length if min_chars is None else min(min_chars, length)
        histogram[bucket_index(length, buckets)] += 1

        if seen_text is not None:
            key = digest64(normalized.casefold())
            if key in seen_text:
                duplicate_texts += 1
            else:
                seen_text.add(key)
        if source_id:
            seen_sources.add(digest64(source_id))

    unique_texts = None if seen_text is None else len(seen_text)
    return {
        "path": str(path),
        "format": "jsonl" if is_jsonl else "text",
        "rows": rows,
        "rows_with_text": rows_with_text,
        "rows_with_empty_text": empty_text,
        "missing_fields": missing_field_counts if is_jsonl else None,
        "characters": total_chars,
        "whitespace_characters": whitespace_chars,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "mean_chars": round(total_chars / max(1, rows_with_text), 2),
        "unique_source_ids": len(seen_sources),
        "unique_texts": unique_texts,
        "duplicate_text_rows": duplicate_texts,
        "duplicate_rate": round(duplicate_texts / max(1, rows_with_text), 6),
        "length_histogram": {
            "buckets": [f"0-{buckets[0]}"] + [f"{buckets[i - 1] + 1}-{buckets[i]}" for i in range(1, len(buckets))]
            + [f">{buckets[-1]}"],
            "counts": histogram,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Plain-text or source-aware JSONL corpora.")
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON report here.")
    parser.add_argument(
        "--tokens-per-character",
        type=float,
        default=0.25,
        help="Token budget estimator. 0.25 means ~4 characters per token "
        "(Indonesian default). Replace with tokens_per_character from "
        "evaluate_tokenizer_corpus.py once a tokenizer exists.",
    )
    parser.add_argument(
        "--token-budget",
        type=float,
        default=None,
        help="Target token count; the report states whether each input reaches it.",
    )
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip exact-duplicate tracking to reduce memory on very large inputs.",
    )
    args = parser.parse_args()

    if args.tokens_per_character <= 0:
        parser.error("--tokens-per-character must be > 0")

    reports = []
    for path in args.inputs:
        if not path.is_file():
            raise SystemExit(f"Input not found: {path}")
        print(f"[corpus_stats] scanning {path} ...", file=sys.stderr)
        reports.append(analyze(path, skip_dedup=args.skip_dedup))

    for report in reports:
        report["estimated_tokens"] = int(round(report["characters"] * args.tokens_per_character))
        if args.token_budget is not None:
            report["reaches_token_budget"] = bool(report["estimated_tokens"] >= args.token_budget)

    combined = {
        "inputs": len(reports),
        "rows": sum(item["rows"] for item in reports),
        "rows_with_text": sum(item["rows_with_text"] for item in reports),
        "characters": sum(item["characters"] for item in reports),
        "estimated_tokens": sum(item["estimated_tokens"] for item in reports),
        "duplicate_text_rows": sum(item["duplicate_text_rows"] for item in reports),
        "tokens_per_character_used": args.tokens_per_character,
        "token_budget": args.token_budget,
    }
    if args.token_budget is not None:
        combined["reaches_token_budget"] = bool(combined["estimated_tokens"] >= args.token_budget)

    document = {"combined": combined, "inputs": reports}
    rendered = json.dumps(document, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
