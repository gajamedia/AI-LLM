#!/usr/bin/env python3
"""Download a pinned Hugging Face CSV and export a reproducible row sample."""

import argparse
import csv
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--license", required=True)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-csv", type=Path,
                        help="Use an already-downloaded CSV instead of fetching it")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_rows < 1:
        parser.error("--max-rows must be positive")

    raw = args.input_csv or args.output.with_suffix(".source.csv")
    if args.input_csv:
        if not raw.is_file():
            raise FileNotFoundError(raw)
    else:
        raw.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{args.dataset}/resolve/{args.revision}/{args.file}?download=true"
        request = urllib.request.Request(url, headers={"User-Agent": "hai07-corpus-import/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, raw.open("wb") as target:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)

    reservoir = []
    scanned = skipped = eligible = 0
    rng = random.Random(args.seed)
    with raw.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or args.text_column not in reader.fieldnames:
            raise ValueError(f"Column {args.text_column!r} missing; found {reader.fieldnames}")
        columns = list(reader.fieldnames)
        for index, source_row in enumerate(reader):
            scanned += 1
            value = source_row.get(args.text_column)
            if not isinstance(value, str):
                skipped += 1
                continue
            text = re.sub(r"\s+", " ", value).strip()
            if len(text) < 40:
                skipped += 1
                continue
            eligible += 1
            metadata = {key: val for key, val in source_row.items() if key != args.text_column and val}
            row = {
                "text": text,
                "source_id": f"hf:{args.dataset}@{args.revision}:{args.file}:row:{index}",
                "source_title": f"{args.dataset} {args.file} row {index}",
                "source_url": f"https://huggingface.co/datasets/{args.dataset}/blob/{args.revision}/{args.file}",
                "license": args.license,
                "language": "id",
                "metadata": {"dataset": args.dataset, "revision": args.revision,
                             "file": args.file, "row_index": index, **metadata},
            }
            item = (index, row)
            if len(reservoir) < args.max_rows:
                reservoir.append(item)
            else:
                replacement = rng.randrange(eligible)
                if replacement < args.max_rows:
                    reservoir[replacement] = item

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reservoir.sort(key=lambda pair: pair[0])
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for _, row in reservoir:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": args.dataset,
        "revision": args.revision,
        "file": args.file,
        "license": args.license,
        "text_column": args.text_column,
        "columns": columns,
        "rows_scanned": scanned,
        "rows_eligible_after_text_filter": eligible,
        "rows_skipped": skipped,
        "rows_written": len(reservoir),
        "sampling": "reservoir sample across complete CSV, sorted by source row index",
        "seed": args.seed,
        "source_path": str(raw),
        "source_bytes": raw.stat().st_size,
        "source_sha256": sha256(raw),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
