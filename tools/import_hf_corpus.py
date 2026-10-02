#!/usr/bin/env python3
"""Stream a Hugging Face text dataset into provenance-aware corpus JSONL.

This importer deliberately requires the dataset ID, revision, text column, and
license to be supplied explicitly. Inspect the upstream dataset card first.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repo ID")
    parser.add_argument("--revision", required=True, help="Immutable commit SHA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--license", required=True, help="License stated by reviewed dataset card")
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow execution of a dataset loading script from the pinned revision")
    args = parser.parse_args()
    if args.max_rows < 1:
        parser.error("--max-rows must be positive")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install tools/requirements-hf-corpus.txt in the active Python environment") from exc

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
        streaming=True,
        trust_remote_code=args.trust_remote_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = skipped = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for index, item in enumerate(dataset):
            if index >= args.max_rows:
                break
            value = item.get(args.text_column)
            if not isinstance(value, str):
                skipped += 1
                continue
            text = re.sub(r"\s+", " ", value).strip()
            if len(text) < 40:
                skipped += 1
                continue
            row = {
                "text": text,
                "source_id": f"hf:{args.dataset}@{args.revision}:{args.split}:{index}",
                "source_title": f"{args.dataset} record {index}",
                "source_url": f"https://huggingface.co/datasets/{args.dataset}/tree/{args.revision}",
                "license": args.license,
                "language": "id",
                "metadata": {"dataset": args.dataset, "revision": args.revision,
                             "split": args.split, "record_index": index},
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    manifest = {
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "text_column": args.text_column,
        "license": args.license,
        "max_rows_scanned": args.max_rows,
        "rows_written": rows,
        "rows_skipped": skipped,
        "output": str(args.output),
        "sha256": sha256(args.output),
        "streaming": True,
        "trust_remote_code": args.trust_remote_code,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
