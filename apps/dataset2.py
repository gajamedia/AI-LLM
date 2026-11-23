import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from config import CONFIG


def _ensure_document_index(token_file, token_data, dtype, bos_token_id, eos_token_id, chunk_tokens=4_000_000):
    """Rebuild document offsets from BOS/EOS markers for legacy token streams."""
    index_path = str(token_file) + ".docs.npy"
    if os.path.exists(index_path) or bos_token_id is None or eos_token_id is None:
        return index_path if os.path.exists(index_path) else None

    bos_count = eos_count = 0
    for offset in range(0, len(token_data), chunk_tokens):
        chunk = token_data[offset:offset + chunk_tokens]
        bos_count += int(np.count_nonzero(chunk == int(bos_token_id)))
        eos_count += int(np.count_nonzero(chunk == int(eos_token_id)))
    if bos_count != eos_count or bos_count == 0:
        raise ValueError(
            f"Cannot infer document boundaries from {token_file}: "
            f"BOS count={bos_count}, EOS count={eos_count}."
        )

    temporary_path = index_path + ".tmp.npy"
    offsets = np.lib.format.open_memmap(
        temporary_path, mode="w+", dtype=np.int64, shape=(eos_count, 2)
    )
    written = 0
    pending_start = None
    try:
        for base in range(0, len(token_data), chunk_tokens):
            chunk = token_data[base:base + chunk_tokens]
            bos_positions = np.flatnonzero(chunk == int(bos_token_id)) + base
            eos_positions = np.flatnonzero(chunk == int(eos_token_id)) + base
            event_positions = np.concatenate((bos_positions, eos_positions))
            event_types = np.concatenate((np.zeros(len(bos_positions), dtype=np.uint8), np.ones(len(eos_positions), dtype=np.uint8)))
            if not len(event_positions):
                continue
            order = np.argsort(event_positions, kind="stable")
            event_positions = event_positions[order]
            event_types = event_types[order]

            cursor = 0
            if pending_start is not None:
                if event_types[0] != 1:
                    raise ValueError("Malformed token stream: BOS encountered before prior EOS.")
                offsets[written] = (pending_start, int(event_positions[0]) + 1)
                written += 1
                cursor = 1
                pending_start = None

            remaining_types = event_types[cursor:]
            remaining_positions = event_positions[cursor:]
            pair_count = len(remaining_types) // 2
            if pair_count:
                expected_bos = remaining_types[:pair_count * 2:2] == 0
                expected_eos = remaining_types[1:pair_count * 2:2] == 1
                if not bool(np.all(expected_bos & expected_eos)):
                    raise ValueError("Malformed token stream: BOS/EOS markers are not properly paired.")
                offsets[written:written + pair_count, 0] = remaining_positions[:pair_count * 2:2]
                offsets[written:written + pair_count, 1] = remaining_positions[1:pair_count * 2:2] + 1
                written += pair_count
            if len(remaining_types) % 2:
                if remaining_types[-1] != 0:
                    raise ValueError("Malformed token stream: EOS has no matching BOS.")
                pending_start = int(remaining_positions[-1])

        if pending_start is not None or written != eos_count:
            raise ValueError(
                f"Incomplete document index for {token_file}: wrote {written} of {eos_count} documents."
            )
        offsets.flush()
        del offsets
        os.replace(temporary_path, index_path)
    except Exception:
        del offsets
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    print(f"Rebuilt document boundary index: {index_path} ({eos_count:,} documents)")
    return index_path


def _resolve_np_dtype(dtype_value):
    if dtype_value is None:
        return np.int32
    if isinstance(dtype_value, str):
        return np.dtype(dtype_value)
    return np.dtype(dtype_value)


# =========================================================
# 1️⃣ PRETOKENIZE (Jalankan SEKALI saja sebelum training)
# =========================================================

def pretokenize_corpus(
    corpus_path,
    tokenizer,
    output_path=None,
    dtype=None,
    log_interval=None,
    chunk_size=100000,
    time_eta=False,
):
    """
    Convert corpus.txt → tokens.bin (int32) with memory-efficient streaming.

    Memory optimization:
    - Streams token IDs directly to file in chunks, NOT collecting all in a list.
      This prevents OOM for large corpora (GB+).
    - chunk_size: number of lines to buffer before flush to disk (default 100k).

    Args:
        corpus_path: Path to input corpus .txt file
        tokenizer: HybridTokenizerLLM instance
        output_path: Path to output .bin file
        dtype: numpy dtype for output (default: int32)
        log_interval: Log progress every N lines
        chunk_size: Buffer size before flush to disk
        time_eta: Show estimated time remaining (requires tqdm or similar)
    """

    output_path = output_path or str(getattr(CONFIG, "PRETOKENIZE_OUTPUT_PATH", "tokens.bin"))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dtype = _resolve_np_dtype(dtype or getattr(CONFIG, "PRETOKENIZE_DTYPE", np.int32))
    log_interval = int(log_interval if log_interval is not None else getattr(CONFIG, "PRETOKENIZE_LOG_INTERVAL", 10000))
    chunk_size = max(1, int(chunk_size))

    # Count total lines for ETA
    total_lines = 0
    if time_eta:
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
        except Exception:
            total_lines = 0

    print("Starting pretokenization (streaming mode)...")
    print(f"Output: {output_path}")
    if total_lines:
        print(f"Total lines: {total_lines}")

    total_tokens = 0
    line_count = 0
    document_offsets = []
    buffer = []
    start_time = None

    with open(corpus_path, "r", encoding="utf-8") as f_in:
        # Open output file in binary write mode for streaming
        with open(output_path, "wb") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    line_count += 1
                    continue

                ids = tokenizer.encode_ids(line, add_bos=True, add_eos=True)
                document_start = total_tokens
                buffer.extend(ids)
                line_count += 1
                total_tokens += len(ids)
                document_offsets.append((document_start, total_tokens))

                # Flush buffer to disk when chunk size reached
                if len(buffer) >= chunk_size:
                    arr = np.array(buffer, dtype=dtype)
                    f_out.write(arr.tobytes())
                    buffer = []

                # Progress logging with optional ETA
                if line_count % log_interval == 0:
                    msg = f"Processed {line_count} lines, {total_tokens} tokens"
                    if total_lines and time_eta:
                        pct = (line_count / total_lines) * 100
                        msg += f" ({pct:.1f}%)"
                    print(msg)

            # Flush remaining buffer after all lines processed
            if buffer:
                arr = np.array(buffer, dtype=dtype)
                f_out.write(arr.tobytes())
                buffer = []

    # Keep document boundaries alongside the flat token stream. Dataset readers
    # use this index to prevent attention targets from crossing documents.
    offsets_path = str(output_path) + ".docs.npy"
    np.save(offsets_path, np.asarray(document_offsets, dtype=np.int64).reshape(-1, 2))

    # Final verification
    final_arr = np.fromfile(output_path, dtype=dtype)
    print(f"Pretokenization finished.")
    print(f"Saved to: {output_path}")
    print(f"Total lines: {line_count}")
    print(f"Total tokens: {len(final_arr)}")
    print(f"Document boundaries: {offsets_path}")


# =========================================================
# 2️⃣ PACKED DATASET (LLM STANDARD)
# =========================================================

class PackedDataset(Dataset):
    """
    Production-level dataset using memory-mapped token file.
    No padding.
    Auto label shifting.
    """

    def __init__(
        self,
        token_file,
        block_size=None,
        dtype=None,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=0,
        document_range=None,
    ):
        assert os.path.exists(token_file), f"{token_file} not found"

        self.block_size = int(block_size if block_size is not None else getattr(CONFIG, "DATASET_BLOCK_SIZE", 128))
        self.pad_token_id = int(pad_token_id)
        dtype = _resolve_np_dtype(dtype or getattr(CONFIG, "DATASET_DTYPE", np.int32))
        self.data = np.memmap(token_file, dtype=dtype, mode="r")
        self.document_offsets_path = _ensure_document_index(
            token_file, self.data, dtype, bos_token_id, eos_token_id
        )
        self.document_offsets = None
        self.sample_spans = None
        if self.document_offsets_path and os.path.exists(self.document_offsets_path):
            self.document_offsets = np.load(self.document_offsets_path, mmap_mode="r", allow_pickle=False)
            if document_range is not None:
                # potong daftar dokumen (mis. sisihkan dokumen terakhir sebagai validasi); token file tetap utuh
                lo, hi = document_range
                self.document_offsets = self.document_offsets[int(lo or 0):(None if hi is None else int(hi))]
                if len(self.document_offsets) == 0:
                    raise ValueError(f"document_range {document_range} kosong untuk {token_file}")
            if len(self.document_offsets) == 0:
                self.document_offsets = None
            else:
                self.sample_spans = self._build_packed_spans(self.document_offsets, self.block_size)
        elif document_range is not None:
            raise ValueError("document_range memerlukan berkas indeks dokumen (.docs.npy).")

        self.total_tokens = len(self.data)
        self.num_blocks = len(self.sample_spans) if self.sample_spans is not None else (self.total_tokens - 1) // self.block_size
        if self.document_offsets is not None:
            self.num_training_tokens = int(
                np.maximum(0, self.document_offsets[:, 1] - self.document_offsets[:, 0] - 1).sum()
            )
        else:
            self.num_training_tokens = max(0, self.total_tokens - 1)

        print("Loaded token file:", token_file)
        print("Total tokens:", self.total_tokens)
        print("Total blocks:", self.num_blocks)

    @staticmethod
    def _build_packed_spans(document_offsets, block_size):
        """Greedily pack whole short documents; split only documents longer than a block."""
        spans = []
        pack_start = pack_end = None
        packed_tokens = 0

        def flush_pack():
            nonlocal pack_start, pack_end, packed_tokens
            if pack_start is not None:
                spans.append((pack_start, pack_end))
            pack_start = pack_end = None
            packed_tokens = 0

        for raw_start, raw_end in document_offsets:
            start, end = int(raw_start), int(raw_end)
            length = end - start
            if length <= 0:
                continue
            if length > block_size:
                flush_pack()
                cursor = start
                # Leave at least one token for the final lookahead target.
                while end - cursor > block_size:
                    spans.append((cursor, cursor + block_size))
                    cursor += block_size
                remainder = end - cursor
                if remainder > 1:
                    pack_start, pack_end, packed_tokens = cursor, end, remainder
                continue
            if packed_tokens + length > block_size:
                flush_pack()
            if pack_start is None:
                pack_start = start
            pack_end = end
            packed_tokens += length

        flush_pack()
        return np.asarray(spans, dtype=np.int64).reshape(-1, 2)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        if self.document_offsets is not None:
            start, end = (int(value) for value in self.sample_spans[idx])
            valid = end - start
            chunk = self.data[start:end]
            input_np = np.full(self.block_size, self.pad_token_id, dtype=np.int64)
            input_np[:valid] = chunk
            labels_np = np.full(self.block_size, -100, dtype=np.int64)
            next_end = min(end + 1, self.total_tokens)
            next_tokens = self.data[start + 1:next_end]
            labels_np[:len(next_tokens)] = next_tokens

            token_positions = np.arange(start, end, dtype=np.int64)
            doc_indices = np.searchsorted(self.document_offsets[:, 1], token_positions, side="right")
            doc_ends = self.document_offsets[doc_indices, 1]
            eos_source_positions = np.flatnonzero(token_positions == (doc_ends - 1))
            labels_np[eos_source_positions] = -100
            local_doc_ids = doc_indices - doc_indices[0]
            document_ids = np.full(self.block_size, -1, dtype=np.int64)
            document_ids[:valid] = local_doc_ids

            input_ids = torch.from_numpy(input_np)
            labels = torch.from_numpy(labels_np)
            document_ids = torch.from_numpy(document_ids)
            return {"input_ids": input_ids, "labels": labels, "document_ids": document_ids}

        start = idx * self.block_size
        end = start + self.block_size + 1

        chunk = self.data[start:end]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return {
            "input_ids": x,
            "labels": y,
        }


def parse_token_spec(spec):
    """'a.u16,b.u16@100:200' -> [('a.u16', None), ('b.u16', (100, 200))].
    '@lo:hi' memilih dokumen lo..hi-1 pada berkas itu (kosong = sampai ujung); koma memisahkan beberapa shard."""
    parts = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        rng = None
        if "@" in item:
            item, r = item.rsplit("@", 1)
            lo, hi = r.split(":")
            rng = (int(lo) if lo else 0, int(hi) if hi else None)
        parts.append((item, rng))
    if not parts:
        raise ValueError("spesifikasi token kosong")
    return parts


class ShardedPackedDataset(Dataset):
    """Beberapa shard token (masing-masing dengan indeks dokumen) tampil sebagai satu dataset berurutan."""

    def __init__(self, parts, block_size=None, dtype=None, bos_token_id=None, eos_token_id=None, pad_token_id=0):
        self.parts = [
            PackedDataset(path, block_size=block_size, dtype=dtype, bos_token_id=bos_token_id,
                          eos_token_id=eos_token_id, pad_token_id=pad_token_id, document_range=rng)
            for path, rng in parts
        ]
        self.block_size = self.parts[0].block_size
        self.cum = np.cumsum([0] + [len(p) for p in self.parts]).astype(np.int64)
        self.total_tokens = sum(p.total_tokens for p in self.parts)
        self.num_training_tokens = sum(p.num_training_tokens for p in self.parts)
        print(f"Sharded dataset: {len(self.parts)} shard, {int(self.cum[-1]):,} blok, {self.num_training_tokens:,} token latih")

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, idx):
        idx = int(idx)
        if not 0 <= idx < int(self.cum[-1]):
            raise IndexError(idx)
        k = int(np.searchsorted(self.cum, idx, side="right")) - 1
        return self.parts[k][idx - int(self.cum[k])]


# =========================================================
# 3️⃣ SLIDING WINDOW DATASET (Optional)
# =========================================================

class SlidingWindowDataset(Dataset):
    """
    Lebih fleksibel tapi lebih mahal.
    """

    def __init__(
        self,
        token_file,
        block_size=None,
        stride=None,
        dtype=None,
    ):
        self.block_size = int(block_size if block_size is not None else getattr(CONFIG, "DATASET_BLOCK_SIZE", 128))
        self.stride = int(stride if stride is not None else getattr(CONFIG, "SLIDING_STRIDE", 1))
        dtype = _resolve_np_dtype(dtype or getattr(CONFIG, "DATASET_DTYPE", np.int32))
        self.data = np.memmap(token_file, dtype=dtype, mode="r")

        self.total_tokens = len(self.data)
        self.num_samples = max(
            0,
            ((self.total_tokens - self.block_size - 1) // self.stride) + 1,
        )
        self.num_training_tokens = self.num_samples * self.block_size

        print("Sliding Dataset loaded")
        print("Total samples:", self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.block_size + 1

        chunk = self.data[start:end]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return {
            "input_ids": x,
            "labels": y,
        }


class ResumableRandomSampler(Sampler):
    """Random sampler that checkpoints its permutation and consumed position."""

    def __init__(self, data_source, shuffle=True, generator=None, rank=0, world_size=1):
        # Multi-GPU (DDP): semua rank memakai permutasi yang SAMA (seed sama); rank r mengambil posisi
        # r, r+W, r+2W, ... sehingga data antar rank tidak tumpang tindih. world_size=1 = perilaku lama.
        self.data_source = data_source
        self.shuffle = bool(shuffle)
        self.generator = generator or torch.Generator()
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank harus di dalam [0, world_size).")
        self.permutation = None
        self.consumed_position = 0   # posisi GLOBAL pada permutasi (semua rank bersama)
        self.yielded_position = 0

    def __len__(self):
        return len(self.data_source) // self.world_size if self.world_size > 1 else len(self.data_source)

    def _ensure_permutation(self):
        if self.permutation is None or len(self.permutation) != len(self.data_source):
            if self.shuffle:
                self.permutation = torch.randperm(len(self.data_source), generator=self.generator)
            else:
                self.permutation = torch.arange(len(self.data_source))
            self.consumed_position = 0
            self.yielded_position = 0

    def __iter__(self):
        self._ensure_permutation()
        self.yielded_position = self.consumed_position
        total = len(self.permutation)
        if self.world_size > 1:
            total -= total % self.world_size   # buang sisa agar semua rank punya jumlah sampel sama
        while self.yielded_position + self.world_size <= total:
            index = int(self.permutation[self.yielded_position + self.rank])
            self.yielded_position += self.world_size
            yield index

    def mark_consumed(self, count):
        self.consumed_position = min(
            self.yielded_position,
            self.consumed_position + int(count) * self.world_size,
        )

    def finish_epoch(self):
        self.permutation = None
        self.consumed_position = 0
        self.yielded_position = 0

    def state_dict(self):
        return {
            "shuffle": self.shuffle,
            "permutation": self.permutation.clone() if self.permutation is not None else None,
            "consumed_position": int(self.consumed_position),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        if bool(state.get("shuffle", self.shuffle)) != self.shuffle:
            raise ValueError("Sampler shuffle mode differs from checkpoint.")
        permutation = state.get("permutation")
        if permutation is not None:
            permutation = torch.as_tensor(permutation, dtype=torch.long, device="cpu")
            if permutation.numel() != len(self.data_source):
                raise ValueError("Sampler checkpoint dataset size differs from current dataset.")
            self.permutation = permutation
        else:
            self.permutation = None
        self.consumed_position = int(state.get("consumed_position", 0))
        self.yielded_position = self.consumed_position
        if not 0 <= self.consumed_position <= len(self.data_source):
            raise ValueError("Sampler checkpoint position is outside the dataset.")
        generator_state = state.get("generator_state")
        if generator_state is not None:
            self.generator.set_state(generator_state)


# =========================================================
# 4️⃣ DATALOADER FACTORY
# =========================================================

def create_dataloader(
    token_file=None,
    block_size=None,
    batch_size=None,
    shuffle=None,
    num_workers=None,
    pin_memory=None,
    persistent_workers=None,
    drop_last=None,
    dataset_type=None,
    stride=None,
    dtype=None,
    resumable=False,
    seed=None,
    rank=0,
    world_size=1,
    bos_token_id=None,
    eos_token_id=None,
    pad_token_id=0,
):
    """
    dataset_type: "packed" or "sliding"
    """

    token_file = token_file or str(getattr(CONFIG, "DATASET_TOKEN_FILE", "tokens.bin"))
    block_size = int(block_size if block_size is not None else getattr(CONFIG, "DATASET_BLOCK_SIZE", 128))
    batch_size = int(batch_size if batch_size is not None else getattr(CONFIG, "DATASET_BATCH_SIZE", 16))
    shuffle = bool(shuffle if shuffle is not None else getattr(CONFIG, "DATASET_SHUFFLE", True))
    num_workers = int(num_workers if num_workers is not None else getattr(CONFIG, "DATASET_NUM_WORKERS", 0))
    pin_memory = bool(pin_memory if pin_memory is not None else getattr(CONFIG, "DATASET_PIN_MEMORY", True))
    persistent_workers = bool(
        persistent_workers if persistent_workers is not None else getattr(CONFIG, "DATASET_PERSISTENT_WORKERS", False)
    )
    drop_last = bool(drop_last if drop_last is not None else getattr(CONFIG, "DATASET_DROP_LAST", False))
    if num_workers == 0:
        persistent_workers = False

    dataset_type = str(dataset_type or getattr(CONFIG, "DATASET_TYPE", "packed")).lower()
    stride = int(stride if stride is not None else getattr(CONFIG, "SLIDING_STRIDE", 1))
    dtype = _resolve_np_dtype(dtype or getattr(CONFIG, "DATASET_DTYPE", np.int32))

    if dataset_type == "packed":
        parts = parse_token_spec(token_file) if ("," in str(token_file) or "@" in str(token_file)) else None
        if parts is not None:
            dataset = ShardedPackedDataset(
                parts, block_size=block_size, dtype=dtype, bos_token_id=bos_token_id,
                eos_token_id=eos_token_id, pad_token_id=pad_token_id,
            )
        else:
            dataset = PackedDataset(
                token_file=token_file,
                block_size=block_size,
                dtype=dtype,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
    elif dataset_type == "sliding":
        dataset = SlidingWindowDataset(
            token_file=token_file,
            block_size=block_size,
            stride=stride,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'packed' or 'sliding'.")

    sampler = None
    if resumable:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
        sampler = ResumableRandomSampler(
            dataset, shuffle=shuffle, generator=generator, rank=rank, world_size=world_size,
        )
    elif int(world_size) > 1:
        raise ValueError("world_size > 1 memerlukan resumable=True (sampler sadar rank).")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    return loader
