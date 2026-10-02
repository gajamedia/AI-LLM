#!/usr/bin/env python3
"""Download one pinned Hugging Face Parquet shard and export bounded JSONL."""

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face commit SHA")
    parser.add_argument("--file", required=True, help="Dataset-relative Parquet shard path")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--license", required=True, help="License declared by reviewed dataset card")
    parser.add_argument("--max-rows", type=int, default=10_000, help="Maximum output rows, sampled across the full shard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-parquet", type=Path,
                        help="Use an already-downloaded shard rather than fetching it again")
    parser.add_argument("--filter", action="append", default=[], metavar="COLUMN=VALUE",
                        help="Keep rows matching this exact metadata value; repeat to require multiple matches")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_rows < 1:
        parser.error("--max-rows must be positive")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install pyarrow in the active Python environment") from exc

    raw = args.input_parquet or args.output.with_suffix(".source.parquet")
    if args.input_parquet:
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

    parquet = pq.ParquetFile(raw)
    filters = {}
    for item in args.filter:
        if "=" not in item:
            parser.error(f"Invalid --filter {item!r}; use COLUMN=VALUE")
        key, value = item.split("=", 1)
        filters[key] = value
    selected_columns = list(parquet.schema_arrow.names)
    if args.text_column not in selected_columns:
        raise ValueError(f"Column {args.text_column!r} missing; found {selected_columns}")
    missing_filters = set(filters) - set(selected_columns)
    if missing_filters:
        raise ValueError(f"Filter columns missing: {sorted(missing_filters)}; found {selected_columns}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    skipped = scanned = matched = eligible = 0
    reservoir = []
    rng = random.Random(args.seed)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        # Read and reservoir-sample the complete shard so output rows are not
        # silently biased toward its first source/domain block.
        for batch in parquet.iter_batches(batch_size=1024, columns=selected_columns):
            columns = {name: batch.column(i).to_pylist() for i, name in enumerate(selected_columns)}
            for offset in range(batch.num_rows):
                index = scanned
                scanned += 1
                original = {name: values[offset] for name, values in columns.items()}
                if any(str(original.get(key, "")) != expected for key, expected in filters.items()):
                    continue
                matched += 1
                value = original.get(args.text_column)
                if not isinstance(value, str):
                    skipped += 1
                    continue
                text = re.sub(r"\s+", " ", value).strip()
                if len(text) < 40:
                    skipped += 1
                    continue
                eligible += 1
                row_meta = {key: val for key, val in original.items() if key != args.text_column and val is not None}
                source_name = str(original.get("source") or args.dataset)
                source_url = str(original.get("url") or
                                 f"https://huggingface.co/datasets/{args.dataset}/blob/{args.revision}/{args.file}")
                row = {
                    "text": text,
                    "source_id": f"hf:{args.dataset}:{source_name}:{hashlib.sha256(source_url.encode('utf-8')).hexdigest()[:16]}",
                    "source_title": source_url,
                    "source_url": source_url,
                    "license": args.license,
                    "language": "id",
                    "metadata": {"dataset": args.dataset, "revision": args.revision,
                                 "file": args.file, "row_index": index, **row_meta},
                }
                item = (index, row)
                if len(reservoir) < args.max_rows:
                    reservoir.append(item)
                else:
                    replacement = rng.randrange(eligible)
                    if replacement < args.max_rows:
                        reservoir[replacement] = item
        reservoir.sort(key=lambda item: item[0])
        for _, row in reservoir:
            output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    manifest = {
        "dataset": args.dataset,
        "revision": args.revision,
        "file": args.file,
        "license": args.license,
        "rows_scanned": scanned,
        "rows_matching_filters": matched,
        "rows_eligible_after_text_filter": eligible,
        "rows_written": len(reservoir),
        "rows_skipped": skipped,
        "filters": filters,
        "sampling": "reservoir sample across entire shard, then sorted by original row index",
        "seed": args.seed,
        "source_bytes": raw.stat().st_size,
        "source_path": str(raw),
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
