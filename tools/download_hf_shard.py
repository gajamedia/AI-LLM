#!/usr/bin/env python3
"""Unduh satu file dataset Hugging Face pada revisi tetap (resumable)."""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--revision", required=True, help="Commit SHA tetap")
    p.add_argument("--file", required=True, help="Path file di dalam repo dataset")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=a.dataset, repo_type="dataset", revision=a.revision,
                           filename=a.file, local_dir=a.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
