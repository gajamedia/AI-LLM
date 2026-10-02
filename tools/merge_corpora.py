#!/usr/bin/env python3
"""Merge source-aware JSONL corpora while deduplicating normalized sentences."""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    output_resolved = args.output.resolve()
    for path in args.inputs:
        if path.resolve() == output_resolved:
            raise SystemExit(
                f"Refusing to merge in place: --output {args.output} is also an input. "
                "The output is opened with mode 'w' first, which would truncate the "
                "input to zero bytes. Write to a new file and rename afterwards."
            )
    seen = set()
    sources = set()
    rows_written = duplicates = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for path in args.inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                        raise ValueError(f"Expected a JSON object with text at {path}:{line_number}")
                    text = re.sub(r"\s+", " ", row["text"]).strip()
                    source_record_id = str(row.get("source_id") or "").strip()
                    title = str(row.get("source_title") or "").strip()
                    url = str(row.get("source_url") or "").strip()
                    if not text or not source_record_id or not title or not url:
                        raise ValueError(f"Missing text/source attribution at {path}:{line_number}")
                    if source_record_id.startswith("idwiki:"):
                        canonical_title = unicodedata.normalize("NFC", title).replace("_", " ")
                        canonical_title = re.sub(r"\s+", " ", canonical_title).strip().casefold()
                        source_id = f"idwiki:title:{canonical_title}"
                    else:
                        source_id = source_record_id
                    key = text.casefold()
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    sources.add(source_id)
                    normalized = {**row, "source_id": source_id, "source_title": title,
                                  "source_record_id": source_record_id,
                                  "source_url": url, "text": text}
                    output.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    rows_written += 1

    manifest = {
        "format": "source-aware-jsonl",
        "rows": rows_written,
        "unique_sources": len(sources),
        "duplicate_rows_removed": duplicates,
        "source_grouping": "canonical Wikipedia titles; non-Wikipedia record IDs preserved",
        "sha256": sha256(args.output),
        "inputs": [{"path": str(path), "rows": sum(1 for line in path.open(encoding="utf-8") if line.strip()),
                    "sha256": sha256(path)} for path in args.inputs],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
