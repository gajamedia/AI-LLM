#!/usr/bin/env python3
"""Filter and deduplicate imported Hugging Face JSONL while preserving provenance."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


ID_STOPWORDS = {
    "ada", "adalah", "agar", "akan", "atau", "bahwa", "bagi", "banyak", "belum",
    "bisa", "dalam", "dan", "dari", "dengan", "di", "ia", "ini", "itu", "juga",
    "karena", "ke", "kita", "lebih", "masih", "mereka", "menjadi", "oleh", "pada",
    "para", "saat", "sebagai", "sehingga", "seperti", "sudah", "telah", "tidak",
    "untuk", "yang",
}
SPAM_HOST_PATTERN = re.compile(
    r"(^|[./_-])(casino|judi|judol|slot|togel|taruhan|sbobet|poker|bandar)([./_-]|$)", re.I
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--include-domain", action="append", default=[],
                        help="Keep only these domains; repeat to allow multiple values")
    parser.add_argument("--include-host", action="append", default=[],
                        help="Keep only these URL host roots and their subdomains")
    parser.add_argument("--min-date", help="Keep ISO-like metadata dates on/after YYYY-MM-DD")
    parser.add_argument("--max-date", help="Keep ISO-like metadata dates on/before YYYY-MM-DD")
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--drop-replacement-char", action="store_true")
    parser.add_argument("--min-indonesian-stopwords", type=int, default=4,
                        help="Require this many Indonesian function words as a basic language filter")
    parser.add_argument("--exclude-obvious-spam-domains", action="store_true")
    args = parser.parse_args()
    if args.max_chars < 40:
        parser.error("--max-chars must be at least 40")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    source_counts = Counter()
    domain_counts = Counter()
    rejection_counts = Counter()
    rows_in = rows_out = 0
    dataset = revision = file = None
    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            rows_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {args.input}:{line_number}") from exc
            text = row.get("text") if isinstance(row, dict) else None
            meta = row.get("metadata", {}) if isinstance(row, dict) else {}
            if not isinstance(text, str) or not isinstance(meta, dict):
                rejection_counts["invalid_schema"] += 1
                continue
            source = str(meta.get("source") or "unknown")
            domain = str(meta.get("domain") or "unknown")
            if source in args.exclude_source:
                rejection_counts[f"excluded_source:{source}"] += 1
                continue
            if args.include_domain and domain not in args.include_domain:
                rejection_counts[f"excluded_domain:{domain}"] += 1
                continue
            row_url = str(row.get("source_url") or "").strip()
            if args.include_host:
                hostname = (urlsplit(row_url).hostname or "").lower()
                if not any(hostname == host.lower() or hostname.endswith("." + host.lower())
                           for host in args.include_host):
                    rejection_counts["excluded_host"] += 1
                    continue
            original_record = meta.get("original_record", {})
            date_value = str(original_record.get("date") or meta.get("date") or "")[:10]
            if args.min_date and (not date_value or date_value < args.min_date):
                rejection_counts["before_min_date_or_missing"] += 1
                continue
            if args.max_date and (not date_value or date_value > args.max_date):
                rejection_counts["after_max_date_or_missing"] += 1
                continue
            normalized = re.sub(r"\s+", " ", text).strip()
            if not normalized:
                rejection_counts["empty"] += 1
                continue
            if args.drop_replacement_char and "\ufffd" in normalized:
                rejection_counts["replacement_character"] += 1
                continue
            if len(normalized) > args.max_chars:
                rejection_counts["too_long"] += 1
                continue
            tokens = re.findall(r"[a-zA-Z]+", normalized.casefold())
            stopword_count = sum(token in ID_STOPWORDS for token in tokens)
            if stopword_count < args.min_indonesian_stopwords:
                rejection_counts["low_indonesian_language_signal"] += 1
                continue
            upstream_url = str(meta.get("url") or "").strip()
            if args.exclude_obvious_spam_domains and upstream_url:
                hostname = urlsplit(upstream_url).hostname or ""
                if SPAM_HOST_PATTERN.search(hostname):
                    rejection_counts["obvious_spam_domain"] += 1
                    continue
            key = normalized.casefold()
            if key in seen:
                rejection_counts["exact_duplicate"] += 1
                continue
            seen.add(key)

            dataset = meta.get("dataset", dataset)
            revision = meta.get("revision", revision)
            file = meta.get("file", file)
            original_url = str(meta.get("upstream_source_url") or original_record.get("url") or "").strip()
            row_index = meta.get("row_index", line_number)
            if original_url:
                group_key = original_url
                title = original_url
            else:
                group_key = f"{source}:record:{row_index}"
                title = f"{source} record {row_index}"
            source_id = f"hf:{dataset}:{source}:{hashlib.sha256(group_key.encode('utf-8')).hexdigest()[:20]}"
            row["text"] = normalized
            row["source_id"] = source_id
            row["source_title"] = title
            # This points to the pinned corpus record; the original page, when
            # present, remains separately recorded in metadata.url.
            row["source_url"] = f"https://huggingface.co/datasets/{dataset}/blob/{revision}/{file}"
            row["metadata"]["upstream_source_url"] = original_url or None
            row["metadata"]["upstream_license_review"] = "required for mixed-source content"
            source_counts[source] += 1
            domain_counts[domain] += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_out += 1

    report = {
        "input": str(args.input),
        "input_sha256": digest(args.input),
        "output": str(args.output),
        "output_sha256": digest(args.output),
        "dataset": dataset,
        "revision": revision,
        "file": file,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_rejected": dict(rejection_counts),
        "output_by_source": dict(source_counts),
        "output_by_domain": dict(domain_counts),
        "max_chars": args.max_chars,
        "excluded_sources": args.exclude_source,
        "included_domains": args.include_domain,
        "included_hosts": args.include_host,
        "min_date": args.min_date,
        "max_date": args.max_date,
        "drop_replacement_char": args.drop_replacement_char,
        "min_indonesian_stopwords": args.min_indonesian_stopwords,
        "exclude_obvious_spam_domains": args.exclude_obvious_spam_domains,
        "exact_deduplication": "normalized whitespace + casefold",
    }
    args.output.with_suffix(args.output.suffix + ".audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
