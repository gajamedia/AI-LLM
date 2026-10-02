#!/usr/bin/env python3
"""Build a small, reproducible multistream XML sample spread across a dump."""

import argparse
import bz2
import hashlib
import json
import random
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_index(path: Path):
    groups = []
    current_offset = None
    current_pages = 0
    with bz2.open(path, "rt", encoding="utf-8") as index:
        for line_number, line in enumerate(index, 1):
            offset_text, page_id, title = line.rstrip("\n").split(":", 2)
            offset = int(offset_text)
            if offset != current_offset:
                if current_offset is not None:
                    groups.append({"offset": current_offset, "pages": current_pages})
                current_offset = offset
                current_pages = 0
            current_pages += 1
    if current_offset is not None:
        groups.append({"offset": current_offset, "pages": current_pages})
    return groups, line_number if "line_number" in locals() else 0


def build_sample(dump: Path, index: Path, output: Path, blocks: int,
                 seed: int, exclude_first_blocks: int = 0) -> dict:
    groups, indexed_pages = read_index(index)
    if blocks < 1 or exclude_first_blocks < 0:
        raise ValueError("blocks must be positive and exclude-first-blocks cannot be negative")
    candidates = groups[exclude_first_blocks:]
    if blocks > len(candidates):
        raise ValueError(f"Requested {blocks} blocks, but only {len(candidates)} are available")
    rng = random.Random(seed)
    selected = sorted(rng.sample(candidates, blocks), key=lambda group: group["offset"])
    next_offset = {groups[i]["offset"]: groups[i + 1]["offset"]
                   for i in range(len(groups) - 1)}
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_size = dump.stat().st_size
    source_bytes = 0
    with dump.open("rb") as source, output.open("wb") as target:
        target.write(bz2.compress(b"<mediawiki>\n", compresslevel=9))
        for group in selected:
            start = group["offset"]
            end = next_offset.get(start, dump_size)
            source.seek(start)
            remaining = end - start
            source_bytes += remaining
            while remaining:
                chunk = source.read(min(4 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError(f"Unexpected end of dump while reading stream at byte {start}")
                target.write(chunk)
                remaining -= len(chunk)
        target.write(bz2.compress(b"\n</mediawiki>", compresslevel=9))

    manifest = {
        "dump_file": str(dump),
        "index_file": str(index),
        "output_file": str(output),
        "index_entries": indexed_pages,
        "available_stream_blocks": len(groups),
        "sample_stream_blocks": blocks,
        "sample_indexed_pages": sum(group["pages"] for group in selected),
        "excluded_first_blocks": exclude_first_blocks,
        "seed": seed,
        "sampled_offsets": [group["offset"] for group in selected],
        "compressed_source_bytes_selected": source_bytes,
        "synthetic_xml_sha256": sha256(output),
        "synthetic_xml_bytes": output.stat().st_size,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=100, help="Index blocks to sample; normally 100 pages each.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-first-blocks", type=int, default=50)
    args = parser.parse_args()
    build_sample(args.dump, args.index, args.output, args.blocks, args.seed, args.exclude_first_blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
