#!/usr/bin/env python3
"""Fast, streaming pretokenizer with trie lookup equivalent to tokenizer2 DP."""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import get_config
from tokenizer2 import build_tokenizer_from_config


def build_trie(tokenizer):
    root = {}
    for token, weight in tokenizer.vocab.items():
        node = root
        for char in token:
            node = node.setdefault(char, {})
        node[None] = (token, math.log(max(float(weight), 1e-12))
                      + tokenizer.length_bonus * (len(token) - 1))
    return root


def encode_tokens_fast(text, tokenizer, trie):
    marked = tokenizer._mark_text(text)
    n = len(marked)
    scores = [float("-inf")] * (n + 1)
    previous = [None] * (n + 1)
    scores[0] = 0.0

    for start in range(n):
        if scores[start] == float("-inf"):
            continue
        node = trie
        end_limit = min(n, start + tokenizer._encode_max_len)
        for end in range(start, end_limit):
            node = node.get(marked[end])
            if node is None:
                break
            terminal = node.get(None)
            if terminal is not None:
                _, token_score = terminal
                next_pos = end + 1
                candidate = scores[start] + token_score
                if candidate > scores[next_pos]:
                    scores[next_pos] = candidate
                    previous[next_pos] = start

    tokens = []
    index = n
    while index > 0:
        prev = previous[index]
        if prev is None:
            tokens.append(marked[index - 1])
            index -= 1
        else:
            tokens.append(marked[prev:index])
            index = prev
    tokens.reverse()
    return tokens


def verify_equivalence(tokenizer, trie, corpus_path, count=1000):
    checked = 0
    with corpus_path.open(encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            expected = tokenizer.encode_tokens(text)
            actual = encode_tokens_fast(text, tokenizer, trie)
            if actual != expected:
                raise RuntimeError(f"Trie encoder differs from standard encoder at sample {checked + 1}.")
            checked += 1
            if checked >= count:
                break
    if not checked:
        raise ValueError(f"No non-empty lines found in {corpus_path}.")
    return checked


def pretokenize(input_path, output_path, tokenizer, trie, dtype, chunk_tokens, log_interval):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    offsets = []
    buffer = []
    total_tokens = 0
    documents = 0
    unknown = 0
    unk_id = tokenizer.token2id[tokenizer.unk_token]
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    with input_path.open(encoding="utf-8") as source, temp_path.open("wb") as target:
        for line in source:
            text = line.strip()
            if not text:
                continue
            ids = [tokenizer.token2id.get(token, unk_id)
                   for token in encode_tokens_fast(text, tokenizer, trie)]
            unknown += sum(token_id == unk_id for token_id in ids)
            ids.insert(0, bos_id)
            ids.append(eos_id)
            start = total_tokens
            total_tokens += len(ids)
            offsets.append((start, total_tokens))
            documents += 1
            buffer.extend(ids)
            if len(buffer) >= chunk_tokens:
                target.write(np.asarray(buffer, dtype=dtype).tobytes())
                buffer.clear()
            if documents % log_interval == 0:
                print(f"Processed {documents:,} documents; {total_tokens:,} tokens", flush=True)
        if buffer:
            target.write(np.asarray(buffer, dtype=dtype).tobytes())

    offset_temp = temp_path.with_name(temp_path.name + ".docs.npy")
    np.save(offset_temp, np.asarray(offsets, dtype=np.int64).reshape(-1, 2))
    os.replace(temp_path, output_path)
    os.replace(offset_temp, Path(str(output_path) + ".docs.npy"))
    byte_count = output_path.stat().st_size
    if byte_count != total_tokens * np.dtype(dtype).itemsize:
        raise RuntimeError("Output byte size does not match token count and dtype.")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "documents": documents,
        "tokens_including_bos_eos": total_tokens,
        "unknown_tokens": unknown,
        "unknown_token_rate": round(unknown / max(1, total_tokens - 2 * documents), 8),
        "dtype": str(np.dtype(dtype)),
        "bytes": byte_count,
        "document_index": str(Path(str(output_path) + ".docs.npy")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="development")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=1000000)
    parser.add_argument("--log-interval", type=int, default=10000)
    parser.add_argument("--verification-samples", type=int, default=1000)
    args = parser.parse_args()

    cfg = get_config(args.env)
    tokenizer = build_tokenizer_from_config(cfg)
    tokenizer.load_all(
        str(args.tokenizer_dir / Path(cfg.VOCAB_PATH).name),
        str(args.tokenizer_dir / Path(cfg.TOKEN2ID_PATH).name),
        str(args.tokenizer_dir / Path(cfg.ID2TOKEN_PATH).name),
    )
    trie = build_trie(tokenizer)
    verified = verify_equivalence(tokenizer, trie, args.train, args.verification_samples)
    print(f"Trie encoder matched tokenizer2 on {verified:,} train documents.", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.int32
    results = {"verified_equivalence_samples": verified, "splits": {}}
    for name, path in (("train", args.train), ("validation", args.validation)):
        results["splits"][name] = pretokenize(
            path,
            args.output_dir / f"{name}_tokens.bin",
            tokenizer,
            trie,
            dtype,
            args.chunk_tokens,
            args.log_interval,
        )
        print(f"Finished {name}: {results['splits'][name]['tokens_including_bos_eos']:,} tokens.", flush=True)

    report_path = args.output_dir / "pretokenization_manifest.json"
    import json
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {report_path}")


if __name__ == "__main__":
    main()
