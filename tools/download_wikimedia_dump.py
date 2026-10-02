#!/usr/bin/env python3
"""Download and checksum a Wikimedia pages-articles multistream dump."""

import argparse
import hashlib
import re
from pathlib import Path

import requests

USER_AGENT = "HAI-LLM corpus builder (local research; sequential download)"
BASE_URL = "https://dumps.wikimedia.org"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki", default="idwiki")
    parser.add_argument("--snapshot", default="20260901")
    parser.add_argument("--kind", choices=("articles", "index"), default="articles",
                        help="Download the article XML or its multistream index.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/corpus_dump_20260901/raw"))
    args = parser.parse_args()

    if args.kind == "index":
        filename = f"{args.wiki}-{args.snapshot}-pages-articles-multistream-index.txt.bz2"
    else:
        filename = f"{args.wiki}-{args.snapshot}-pages-articles-multistream.xml.bz2"
    folder_url = f"{BASE_URL}/{args.wiki}/{args.snapshot}/"
    headers = {"User-Agent": USER_AGENT}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / filename
    partial = target.with_suffix(target.suffix + ".part")

    checksum_url = folder_url + f"{args.wiki}-{args.snapshot}-sha1sums.txt"
    response = requests.get(checksum_url, headers=headers, timeout=(20, 60))
    response.raise_for_status()
    expected = None
    for line in response.text.splitlines():
        match = re.match(r"^([0-9a-fA-F]{40})\s+\*?(.+?)\s*$", line)
        if match and match.group(2) == filename:
            expected = match.group(1).lower()
            break
    if not expected:
        raise RuntimeError(f"Could not find SHA-1 for {filename} in official checksum list")

    if target.exists():
        current_hash = hashlib.sha1()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                current_hash.update(chunk)
        if current_hash.hexdigest() == expected:
            print(f"Already downloaded and verified: {target}")
            return 0
        raise RuntimeError(f"Existing file failed SHA-1; move it aside before retrying: {target}")

    offset = partial.stat().st_size if partial.exists() else 0
    request_headers = dict(headers)
    if offset:
        request_headers["Range"] = f"bytes={offset}-"
    dump_url = folder_url + filename
    with requests.get(dump_url, headers=request_headers, stream=True, timeout=(30, 120)) as dump:
        dump.raise_for_status()
        append = offset > 0 and dump.status_code == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        content_length = int(dump.headers.get("Content-Length", "0"))
        total = offset + content_length if content_length else None
        last_report = 0
        written = offset
        with partial.open(mode) as output:
            for chunk in dump.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                written += len(chunk)
                if written - last_report >= 64 * 1024 * 1024:
                    pct = f" ({written / total:.1%})" if total else ""
                    print(f"Downloaded {written / (1024**3):.2f} GiB{pct}", flush=True)
                    last_report = written

    digest = hashlib.sha1()
    with partial.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA-1 mismatch: expected {expected}, got {actual}; partial file retained")
    partial.replace(target)
    print(f"Verified SHA-1 {actual}; saved {target} ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
