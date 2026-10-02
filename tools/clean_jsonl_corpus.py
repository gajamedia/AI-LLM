#!/usr/bin/env python3
"""Clean provenance-aware corpus JSONL without changing its row schema.

Rows keep every key they arrived with; only `text` is ever rewritten. Records
that fail a quality rule are dropped and counted, never silently discarded, and
every run writes a `.audit.json` next to the output.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

WS_RE = re.compile(r"\s+")
# Wikipedia exposes its categories as trailing junk in the last sentence of a
# page; category tags are metadata, never prose, so the row is cut there.
CATEGORY_RE = re.compile(r"\s*\b(?:Kategori|Category):.*$", re.DOTALL)
# Closing quotes and brackets are not prose endings; strip them before asking
# whether the row actually finishes a sentence. Lists, tables, discography
# dumps and web-page footers are the rows that fail this test.
TRAILING_CLOSERS = "\"'”’»)]"
SENTENCE_ENDINGS = (".", "!", "?", "…")
# Section titles that only ever introduce link/reference blocks. A trailing
# colon means the token was an article heading or title ("Sumber: ..."), not a
# section, so that spelling is deliberately left alone.
BOILERPLATE_HEAD_RE = re.compile(
    r"^(?:"
    r"Lihat\s+pula|Referensi|Catatan\s+kaki|Pranala\s+luar|"
    r"Pranala\s+eksternal|Daftar\s+pustaka|Daftar\s+rujukan|Bacaan\s+lanjut|"
    r"Bacaan\s+tambahan|Galeri|Singkatan|Glosarium|Daftar\s+isi|Rangkuman|"
    r"Ringkasan|Footnote|External\s+links|References|Further\s+reading"
    r")(?!\s*:)(?:\s|$)",
    re.IGNORECASE,
)
LICENSE_BY_PREFIX = (
    ("idwiki:", "cc-by-sa-4.0"),
    ("hf:HPLT/HPLT2.0_cleaned", "cc0-1.0"),
)


def normalize(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def fingerprint(text: str) -> int:
    """64-bit digest of the normalized, case-folded text."""
    digest = hashlib.blake2b(text.casefold().encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def resolve_license(row: dict, fallback: str) -> str | None:
    existing = row.get("license")
    if isinstance(existing, str) and existing.strip():
        return None
    source_id = str(row.get("source_id") or "")
    for prefix, license_id in LICENSE_BY_PREFIX:
        if source_id.startswith(prefix):
            return license_id
    return fallback or None


def trim_to_last_sentence(text: str) -> tuple[str | None, bool]:
    """Cut trailing footer/list junk back to the last finished sentence.

    Returns (kept_text, was_trimmed). A period inside a number is never a
    sentence boundary: it must be followed by whitespace, a closing bracket or
    the end of the row. Returns None when no sentence boundary exists at all.
    """
    if text.rstrip(TRAILING_CLOSERS).endswith(SENTENCE_ENDINGS):
        return text, False
    for position in range(len(text) - 1, -1, -1):
        char = text[position]
        following = text[position + 1] if position + 1 < len(text) else ""
        if char == "…":
            return text[: position + 1], True
        if char in ".!?":
            if following == "" or following.isspace() or following in TRAILING_CLOSERS:
                return text[: position + 1].rstrip(TRAILING_CLOSERS), True
    return None, True


def clean_row(row: dict, args) -> tuple[str | None, str | None]:
    """Return (kept_text, rejection_reason)."""
    text = row.get("text")
    if not isinstance(text, str):
        return None, "invalid_schema"

    if args.truncate_at_category:
        cut = CATEGORY_RE.search(text)
        if cut:
            text = text[: cut.start()]
    text = normalize(text)

    if args.trim_last_sentence:
        text, trimmed = trim_to_last_sentence(text)
        if text is None:
            return None, "no_sentence_ending"
        if trimmed:
            text = normalize(text)

    if len(text) < args.min_chars:
        return None, "too_short"
    if BOILERPLATE_HEAD_RE.match(text):
        return None, "boilerplate_heading"
    letters = sum(char.isalpha() for char in text)
    if letters / len(text) < args.min_alpha_ratio:
        return None, "too_few_letters"
    digits = sum(char.isdigit() for char in text)
    if digits / len(text) >= args.max_digit_ratio:
        return None, "digit_heavy"
    return text, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path,
                        help="Source JSONL; repeat to clean several files in one pass")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=None,
                        help="Defaults to <output>.audit.json")
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.60)
    parser.add_argument("--max-digit-ratio", type=float, default=0.25)
    parser.add_argument("--truncate-at-category", action="store_true", default=True)
    parser.add_argument("--keep-categories", dest="truncate_at_category",
                        action="store_false", help="Keep Kategori:/Category: text")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false",
                        help="Keep rows whose normalized text already appeared")
    parser.add_argument("--trim-last-sentence", dest="trim_last_sentence",
                        action="store_true", default=True,
                        help="Cut trailing list/table/footer junk back to the last "
                             "finished sentence (default)")
    parser.add_argument("--keep-tail", dest="trim_last_sentence",
                        action="store_false",
                        help="Keep rows verbatim even when they never finish a sentence")
    parser.add_argument("--default-license", default="",
                        help="Applied when the row has no license and no known source prefix")
    args = parser.parse_args()
    if args.min_chars < 1:
        parser.error("--min-chars must be positive")
    if not 0 < args.min_alpha_ratio <= 1:
        parser.error("--min-alpha-ratio must be in (0, 1]")
    if not 0 <= args.max_digit_ratio <= 1:
        parser.error("--max-digit-ratio must be in [0, 1]")

    audit_path = args.audit or args.output.with_suffix(args.output.suffix + ".audit.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[int] = set()
    rejection_counts = Counter()
    input_stats = []
    rows_out = 0
    keys_added = Counter()

    with args.output.open("w", encoding="utf-8", newline="\n") as dst:
        for source in args.input:
            counts = Counter()
            with source.open("r", encoding="utf-8") as src:
                for line_number, line in enumerate(src, 1):
                    if not line.strip():
                        continue
                    counts["rows_in"] += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        rejection_counts["invalid_json"] += 1
                        counts["invalid_json"] += 1
                        continue
                    if not isinstance(row, dict):
                        rejection_counts["invalid_schema"] += 1
                        counts["invalid_schema"] += 1
                        continue

                    kept, reason = clean_row(row, args)
                    if kept is None:
                        rejection_counts[reason] += 1
                        counts[reason] += 1
                        continue
                    if args.dedup:
                        key = fingerprint(kept)
                        if key in seen:
                            rejection_counts["exact_duplicate"] += 1
                            counts["exact_duplicate"] += 1
                            continue
                        seen.add(key)

                    row["text"] = kept
                    license_id = resolve_license(row, args.default_license)
                    if license_id:
                        row["license"] = license_id
                        keys_added["license"] += 1
                    if not str(row.get("language") or "").strip():
                        row["language"] = "id"
                        keys_added["language"] += 1
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_out += 1
                    counts["rows_out"] += 1
            input_stats.append({"input": str(source), **{k: counts[k] for k in sorted(counts)}})

    report = {
        "output": str(args.output),
        "inputs": input_stats,
        "rows_in": sum(item["rows_in"] for item in input_stats),
        "rows_out": rows_out,
        "rows_rejected": dict(rejection_counts),
        "keys_backfilled": dict(keys_added),
        "filters": {
            "min_chars": args.min_chars,
            "min_alpha_ratio": args.min_alpha_ratio,
            "max_digit_ratio": args.max_digit_ratio,
            "truncate_at_category": args.truncate_at_category,
            "trim_last_sentence": args.trim_last_sentence,
            "deduplicated": args.dedup,
            "dedup_key": "blake2b-64 of casefolded normalized text",
            "boilerplate_headings_dropped": True,
        },
        "row_schema_note": "existing keys preserved; only 'text' rewritten",
    }
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
