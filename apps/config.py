# config.py
"""
Configuration file untuk Project HAI-LLM
Menyimpan semua parameter yang dapat dikonfigurasi
"""

import os
from pathlib import Path
from typing import List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ========================================
# Base Configuration
# ========================================
class Config:
    """Base configuration class untuk tokenizer"""

    # Tokenizer Parameters
    MAX_SUB_LEN = 4          # Maksimal panjang substring awal dalam vocab building
    MIN_FREQ = 20             # Minimum frequency untuk token di vocab awal

    # Special Tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    # Training Parameters
    VOCAB_SIZE = 20000         # Target vocabulary size

    # Hybrid Scoring Parameters
    # alpha: BPE-style frequency pressure
    # beta: WordPiece-style association (PMI-like)
    # gamma: Unigram-style stability bonus
    ALPHA = 1.0
    BETA = 0.5
    GAMMA = 0.3

    # Hybrid merge controls
    MIN_PAIR_FREQ = 100        # Ignore low-frequency pairs during merge search
    MAX_TOKEN_LENGTH = None  # None -> tokenizer default (max(8, MAX_SUB_LEN*4))
    MAX_MERGES = None        # None -> tokenizer default (VOCAB_SIZE*2)
    PRUNE_EVERY = 200         # Periodic prune cadence during training
    MERGES_PER_ROUND = 32     # Add top-N merge candidates per scoring pass
    LENGTH_BONUS = 0.2      # Decode-time preference for longer subwords
    NORMALIZE_WHITESPACE = True
    # Memory optimization for large corpora
    MAX_VOCAB_SAMPLES = None    # Max texts to process for initial vocab (None=unlimited)
    MAX_COUNTER_SIZE = None     # Prune counter when exceeding this size (None=no prune)
    PAIR_SUBSAMPLE_RATIO = None # Fraction of corpus for pair scoring (0.0-1.0, None=all)

    # Encoding Parameters
    MAX_LENGTH = None        # Default max_length untuk encoding (None = variable)

    # ========================================
    # Embedding Defaults
    # ========================================
    EMBEDDING_DIM = 256
    EMBEDDING_DROPOUT = 0.1
    POSITIONAL_EMBEDDING_TYPE = "learned"  # "learned", "sinusoidal", or "rope"
    ROPE_BASE = 10000.0
    EMBEDDING_MAX_SEQ_LEN = None  # None -> derive from DATASET_BLOCK_SIZE

    # ========================================
    # Attention Defaults
    # ========================================
    ATTENTION_NUM_HEADS = 8
    ATTENTION_DROPOUT = 0.1
    ATTENTION_USE_BIAS = False
    ATTENTION_MAX_SEQ_LEN = None  # None -> derive from EMBEDDING_MAX_SEQ_LEN / DATASET_BLOCK_SIZE
    ATTENTION_NUM_KV_HEADS = 0    # 0 -> sama dengan NUM_HEADS (MHA mode). >0 -> GQA mode

    # ========================================
    # Transformer Block/Stack Defaults
    # ========================================
    TRANSFORMER_NUM_LAYERS = 6
    TRANSFORMER_MLP_RATIO = 4.0
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_USE_BIAS = False
    NORMALIZATION_TYPE = "rmsnorm" # "layernorm"
    MLP_TYPE = "swiglu" # "gelu"

    # ========================================
    # GPT Training Script Defaults (tools/train_gpt.py)
    # ========================================
    TRAIN_DEVICE = "auto"       # "auto", "cpu", or "cuda"
    TRAIN_EPOCHS = 1
    TRAIN_MAX_STEPS = None      # None -> run full dataloader for all epochs
    TRAIN_LR = 3e-4
    TRAIN_WEIGHT_DECAY = 0.1
    TRAIN_GRAD_CLIP = 1.0       # <= 0 disables clipping
    TRAIN_LOG_EVERY = 20
    TRAIN_SAVE_DIR = "reports/checkpoints"
    TRAIN_SAVE_LOG_PATH = "reports/logs/train_save.log"
    TRAIN_SAVE_EVERY = 0        # 0 disables periodic checkpointing
    TRAIN_LOSS_WINDOW = 50
    TRAIN_PLATEAU_ENABLED = True
    TRAIN_PLATEAU_FACTOR = 0.5
    TRAIN_PLATEAU_PATIENCE = 3
    TRAIN_PLATEAU_THRESHOLD = 1e-4
    TRAIN_PLATEAU_MIN_LR = 1e-6
    TRAIN_PLATEAU_COOLDOWN = 0
    TRAIN_SCHEDULER = "cosine"   # "cosine" | "plateau" | "none" (CLI --plateau/--no-plateau menimpa)
    TRAIN_WARMUP_STEPS = 500     # linear warmup sebelum cosine
    TRAIN_MIN_LR = 1e-5          # LR akhir pada cosine
    TRAIN_AMP_ENABLED = False    # bf16 autocast; hanya aktif bila hardware mendukung
    TRAIN_EARLY_STOP_ENABLED = False
    TRAIN_EARLY_STOP_PATIENCE = 8
    TRAIN_EARLY_STOP_MIN_DELTA = 1e-4
    TRAIN_GRAD_ACCUM = 1  # Gradient accumulation steps

    # ========================================
    # Dataset / DataLoader Defaults
    # ========================================
    DATASET_BATCH_SIZE = 32
    DATASET_SHUFFLE = True
    DATASET_NUM_WORKERS = 0
    DATASET_BLOCK_SIZE = 128
    DATASET_TYPE = "packed"  # "packed" or "sliding"
    DATASET_PIN_MEMORY = True
    DATASET_PERSISTENT_WORKERS = False
    DATASET_DTYPE = np.int32
    DATASET_SHUFFLE_BUFFER_SIZE = None  # if None, shuffle full index
    DATASET_DROP_LAST = False
    # determinism & performance
    DATASET_SEED = None
    DATASET_PREFETCH_SIZE = 0
    DATASET_USE_MULTIPROCESSING = False
    DATASET_FAIL_FAST = False
    # sharding (set at runtime when running distributed)
    DATASET_NUM_REPLICAS = None
    DATASET_RANK = None
    SLIDING_STRIDE = 1
    SLIDING_STREAMING = True
    SLIDING_CACHE_SIZE = 16

    # ========================================
    # File Paths Configuration
    # ========================================
    BASE_DIR = PROJECT_ROOT / "data"

    CORPUS_FILE = BASE_DIR / "corpus.txt"
    DATASET_TOKEN_FILE = BASE_DIR / "tokens.bin"
    VOCAB_PATH = BASE_DIR / "vocab_llm.json"
    TOKEN2ID_PATH = BASE_DIR / "token2id_llm.json"
    ID2TOKEN_PATH = BASE_DIR / "id2token_llm.json"
    PRETOKENIZE_OUTPUT_PATH = DATASET_TOKEN_FILE
    PRETOKENIZE_DTYPE = np.int32
    PRETOKENIZE_LOG_INTERVAL = 10000
    TRAIN_VAL_TOKEN_FILE = None
    REQUIRE_CORPUS_FILE = False

    # ========================================
    # Corpus Configuration
    # ========================================
    DEFAULT_CORPUS = [
        "mengembangkan tokenizer",
        "tokenizer bahasa indonesia",
        "pengembangan model bahasa",
        "tokenizer hybrid production-ready",
        "natural language processing",
        "machine learning model",
    ]

    @classmethod
    def load_corpus(cls) -> List[str]:
        """
        Load corpus dari file corpus.txt.
        Development dan testing fallback ke DEFAULT_CORPUS; production wajib punya corpus file.

        Returns:
            List of corpus texts
        """
        require_corpus = getattr(cls, "REQUIRE_CORPUS_FILE", False)

        try:
            if cls.CORPUS_FILE.exists():
                with open(cls.CORPUS_FILE, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                if lines:
                    print(f"Loaded {len(lines)} lines from {cls.CORPUS_FILE}")
                    return lines

                message = f"File {cls.CORPUS_FILE} is empty"
                if require_corpus:
                    raise ValueError(message)

                print(f"{message}, using DEFAULT_CORPUS")
                return cls.DEFAULT_CORPUS

            message = f"File {cls.CORPUS_FILE} not found"
            if require_corpus:
                raise FileNotFoundError(message)

            print(f"{message}, using DEFAULT_CORPUS")
            return cls.DEFAULT_CORPUS
        except Exception as e:
            if require_corpus:
                raise

            print(f"Error reading corpus file: {e}, using DEFAULT_CORPUS")
            return cls.DEFAULT_CORPUS

    @classmethod
    def get_corpus(cls) -> List[str]:
        """Get corpus (lazy loading)"""
        if not hasattr(cls, "_corpus"):
            cls._corpus = cls.load_corpus()
        return cls._corpus

    @classmethod
    def to_dict(cls):
        """Convert config ke dictionary"""
        return {
            "max_sub_len": cls.MAX_SUB_LEN,
            "min_freq": cls.MIN_FREQ,
            "pad_token": cls.PAD_TOKEN,
            "unk_token": cls.UNK_TOKEN,
            "bos_token": cls.BOS_TOKEN,
            "eos_token": cls.EOS_TOKEN,
            "vocab_size": cls.VOCAB_SIZE,
            "alpha": cls.ALPHA,
            "beta": cls.BETA,
            "gamma": cls.GAMMA,
            "min_pair_freq": cls.MIN_PAIR_FREQ,
            "max_token_length": cls.MAX_TOKEN_LENGTH,
            "max_merges": cls.MAX_MERGES,
            "prune_every": cls.PRUNE_EVERY,
            "merges_per_round": cls.MERGES_PER_ROUND,
            "length_bonus": cls.LENGTH_BONUS,
            "normalize_whitespace": cls.NORMALIZE_WHITESPACE,
            "max_vocab_samples": cls.MAX_VOCAB_SAMPLES,
            "max_counter_size": cls.MAX_COUNTER_SIZE,
            "pair_subsample_ratio": cls.PAIR_SUBSAMPLE_RATIO,
            "max_length": cls.MAX_LENGTH,
            "embedding_dim": cls.EMBEDDING_DIM,
            "embedding_dropout": cls.EMBEDDING_DROPOUT,
            "positional_embedding_type": cls.POSITIONAL_EMBEDDING_TYPE,
            "rope_base": cls.ROPE_BASE,
            "embedding_max_seq_len": cls.EMBEDDING_MAX_SEQ_LEN,
            "attention_num_heads": cls.ATTENTION_NUM_HEADS,
            "attention_dropout": cls.ATTENTION_DROPOUT,
            "attention_use_bias": cls.ATTENTION_USE_BIAS,
            "attention_max_seq_len": cls.ATTENTION_MAX_SEQ_LEN,
            "attention_num_kv_heads": cls.ATTENTION_NUM_KV_HEADS,
            "transformer_num_layers": cls.TRANSFORMER_NUM_LAYERS,
            "transformer_mlp_ratio": cls.TRANSFORMER_MLP_RATIO,
            "transformer_dropout": cls.TRANSFORMER_DROPOUT,
            "transformer_use_bias": cls.TRANSFORMER_USE_BIAS,
            "train_device": cls.TRAIN_DEVICE,
            "train_epochs": cls.TRAIN_EPOCHS,
            "train_max_steps": cls.TRAIN_MAX_STEPS,
            "train_lr": cls.TRAIN_LR,
            "train_weight_decay": cls.TRAIN_WEIGHT_DECAY,
            "train_grad_clip": cls.TRAIN_GRAD_CLIP,
            "train_log_every": cls.TRAIN_LOG_EVERY,
            "train_save_dir": cls.TRAIN_SAVE_DIR,
            "train_save_log_path": cls.TRAIN_SAVE_LOG_PATH,
            "train_save_every": cls.TRAIN_SAVE_EVERY,
            "train_loss_window": cls.TRAIN_LOSS_WINDOW,
            "train_plateau_enabled": cls.TRAIN_PLATEAU_ENABLED,
            "train_plateau_factor": cls.TRAIN_PLATEAU_FACTOR,
            "train_plateau_patience": cls.TRAIN_PLATEAU_PATIENCE,
            "train_plateau_threshold": cls.TRAIN_PLATEAU_THRESHOLD,
            "train_plateau_min_lr": cls.TRAIN_PLATEAU_MIN_LR,
            "train_plateau_cooldown": cls.TRAIN_PLATEAU_COOLDOWN,
            "train_scheduler": cls.TRAIN_SCHEDULER,
            "train_warmup_steps": cls.TRAIN_WARMUP_STEPS,
            "train_min_lr": cls.TRAIN_MIN_LR,
            "train_amp_enabled": cls.TRAIN_AMP_ENABLED,
            "train_early_stop_enabled": cls.TRAIN_EARLY_STOP_ENABLED,
            "train_early_stop_patience": cls.TRAIN_EARLY_STOP_PATIENCE,
            "train_early_stop_min_delta": cls.TRAIN_EARLY_STOP_MIN_DELTA,
            "dataset_batch_size": cls.DATASET_BATCH_SIZE,
            "dataset_shuffle": cls.DATASET_SHUFFLE,
            "dataset_num_workers": cls.DATASET_NUM_WORKERS,
            "dataset_block_size": cls.DATASET_BLOCK_SIZE,
            "dataset_type": cls.DATASET_TYPE,
            "dataset_pin_memory": cls.DATASET_PIN_MEMORY,
            "dataset_persistent_workers": cls.DATASET_PERSISTENT_WORKERS,
            "dataset_dtype": str(np.dtype(cls.DATASET_DTYPE)),
            "dataset_shuffle_buffer_size": cls.DATASET_SHUFFLE_BUFFER_SIZE,
            "dataset_drop_last": cls.DATASET_DROP_LAST,
            "dataset_seed": cls.DATASET_SEED,
            "dataset_prefetch_size": cls.DATASET_PREFETCH_SIZE,
            "dataset_use_multiprocessing": cls.DATASET_USE_MULTIPROCESSING,
            "dataset_fail_fast": cls.DATASET_FAIL_FAST,
            "dataset_num_replicas": cls.DATASET_NUM_REPLICAS,
            "dataset_rank": cls.DATASET_RANK,
            "dataset_token_file": str(cls.DATASET_TOKEN_FILE),
            "sliding_stride": cls.SLIDING_STRIDE,
            "sliding_streaming": cls.SLIDING_STREAMING,
            "sliding_cache_size": cls.SLIDING_CACHE_SIZE,
            "pretokenize_output_path": str(cls.PRETOKENIZE_OUTPUT_PATH),
            "pretokenize_dtype": str(np.dtype(cls.PRETOKENIZE_DTYPE)),
            "pretokenize_log_interval": cls.PRETOKENIZE_LOG_INTERVAL,
            "require_corpus_file": cls.REQUIRE_CORPUS_FILE,
            "train_grad_accum": cls.TRAIN_GRAD_ACCUM,
        }

    @classmethod
    def to_file_paths(cls):
        """Get file paths configuration"""
        return {
            "vocab_path": str(cls.VOCAB_PATH),
            "token2id_path": str(cls.TOKEN2ID_PATH),
            "id2token_path": str(cls.ID2TOKEN_PATH),
        }


# ========================================
# Development Configuration
# ========================================
class DevelopmentConfig(Config):
    """Development settings; the trainer reports token/parameter and epoch budgets at startup."""

    # ========================================
    # Tokenizer — vocab 8000 untuk data 205K kata
    # ========================================
    VOCAB_SIZE = 8000             # 205K kata support 8k vocab (2x dari sebelumnya)
    MIN_FREQ = 5                  # Lebih ketat untuk filter noise
    MAX_SUB_LEN = 5               # Subword lebih panjang
    MIN_PAIR_FREQ = 5
    MAX_TOKEN_LENGTH = 16
    MAX_MERGES = 7996             # VOCAB_SIZE - 4
    PRUNE_EVERY = 300
    MERGES_PER_ROUND = 48
    LENGTH_BONUS = 0.2
    NORMALIZE_WHITESPACE = True
    ALPHA = 1.0
    BETA = 0.5
    GAMMA = 0.3

    # Memory optimization for large corpora
    MAX_VOCAB_SAMPLES = 500000     # Batasi corpus untuk initial vocab (6.2M terlalu besar)
    MAX_COUNTER_SIZE = 5000000     # Prune counter saat exceeding
    PAIR_SUBSAMPLE_RATIO = 0.3    # Gunakan 30% corpus untuk pair scoring

    # ========================================
    # Model — kapasitas lebih besar untuk data lebih banyak
    # ========================================
    MAX_LENGTH = 512              # Konteks lebih panjang
    EMBEDDING_DIM = 384           # +50% dari 256
    EMBEDDING_DROPOUT = 0.1
    POSITIONAL_EMBEDDING_TYPE = "rope"  # Gunakan RoPE untuk generalization lebih baik
    ROPE_BASE = 10000.0
    EMBEDDING_MAX_SEQ_LEN = 512
    ATTENTION_NUM_HEADS = 6       # 384/64 = 6 heads
    ATTENTION_DROPOUT = 0.1
    ATTENTION_USE_BIAS = False
    ATTENTION_MAX_SEQ_LEN = 512
    ATTENTION_NUM_KV_HEADS = 3    # GQA: 3 KV heads melayani 6 query heads (2:1 ratio)
    TRANSFORMER_NUM_LAYERS = 6    # Dari 4 → 6 (lebih dalam)
    TRANSFORMER_MLP_RATIO = 4.0
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_USE_BIAS = False

    # ========================================
    # Training — start with one pass; increase the budget based on validation results.
    # ========================================
    TRAIN_DEVICE = "cpu"
    TRAIN_EPOCHS = 1
    TRAIN_MAX_STEPS = None
    TRAIN_LR = 5e-4               # Naikkan dari 3e-4 (data lebih banyak = bisa agresif)
    TRAIN_WEIGHT_DECAY = 0.1      # Standard
    TRAIN_GRAD_CLIP = 1.0
    TRAIN_LOG_EVERY = 20
    TRAIN_SAVE_DIR = "reports/checkpoints"
    TRAIN_SAVE_EVERY = 500
    TRAIN_LOSS_WINDOW = 50
    TRAIN_PLATEAU_ENABLED = True
    TRAIN_PLATEAU_FACTOR = 0.5
    TRAIN_PLATEAU_PATIENCE = 5    # Dari 8 → 5
    TRAIN_PLATEAU_THRESHOLD = 1e-4
    TRAIN_PLATEAU_MIN_LR = 1e-6
    TRAIN_PLATEAU_COOLDOWN = 2
    TRAIN_SCHEDULER = "cosine"
    TRAIN_WARMUP_STEPS = 100      # run dev pendek; jangan buang seluruh langkah di warmup
    TRAIN_MIN_LR = 1e-5
    TRAIN_EARLY_STOP_ENABLED = True
    TRAIN_EARLY_STOP_PATIENCE = 15  # Dari 25 → 15 (data cukup, tidak perlu terlalu sabar)
    TRAIN_EARLY_STOP_MIN_DELTA = 1e-4
    TRAIN_GRAD_ACCUM = 4          # Effective batch = 16*4 = 64

    # ========================================
    # Dataset — konteks lebih panjang untuk learning lebih baik
    # ========================================
    DATASET_BATCH_SIZE = 16
    DATASET_SHUFFLE = True
    DATASET_NUM_WORKERS = 0
    DATASET_BLOCK_SIZE = 256      # Dari 128 → 256 (konteks 2x lebih panjang)
    DATASET_TYPE = "packed"
    DATASET_PIN_MEMORY = False
    DATASET_DTYPE = np.int32
    DATASET_PERSISTENT_WORKERS = False
    DATASET_PREFETCH_SIZE = 2
    DATASET_USE_MULTIPROCESSING = False
    DATASET_DROP_LAST = True
    DATASET_FAIL_FAST = False
    SLIDING_STRIDE = 1
    SLIDING_STREAMING = False
    SLIDING_CACHE_SIZE = 16

    # ========================================
    # Paths
    # ========================================
    BASE_DIR = PROJECT_ROOT / "data" / "dev"
    CORPUS_FILE = PROJECT_ROOT / "data" / "corpus.txt"
    DATASET_TOKEN_FILE = BASE_DIR / "tokens_train.bin"
    VOCAB_PATH = BASE_DIR / "vocab_dev.json"
    TOKEN2ID_PATH = BASE_DIR / "token2id_dev.json"
    ID2TOKEN_PATH = BASE_DIR / "id2token_dev.json"
    PRETOKENIZE_OUTPUT_PATH = DATASET_TOKEN_FILE
    PRETOKENIZE_DTYPE = np.int32
    PRETOKENIZE_LOG_INTERVAL = 2000
    TRAIN_VAL_TOKEN_FILE = BASE_DIR / "tokens_val.bin"

    DEFAULT_CORPUS = [
        "hello world",
        "tokenizer test",
        "development mode",
    ]


# ========================================
# Production Configuration
# ========================================
class ProductionConfig(Config):
    """Konfigurasi untuk production"""
    REQUIRE_CORPUS_FILE = True
    VOCAB_SIZE = 32768 #20000
    MIN_FREQ = 20
    MAX_SUB_LEN = 6
    MAX_LENGTH = 256 #512
    EMBEDDING_DIM = 384 #768
    EMBEDDING_DROPOUT = 0.1
    POSITIONAL_EMBEDDING_TYPE = "learned"
    ROPE_BASE = 10000.0
    EMBEDDING_MAX_SEQ_LEN = 256 #512
    ATTENTION_NUM_HEADS = 6 #8 #12
    ATTENTION_DROPOUT = 0.1
    ATTENTION_USE_BIAS = False
    ATTENTION_MAX_SEQ_LEN = 256 #512
    ATTENTION_NUM_KV_HEADS = 3  # GQA: 3 KV heads melayani 6 query heads (2:1 ratio)
    TRANSFORMER_NUM_LAYERS = 8 #6 #12
    TRANSFORMER_MLP_RATIO = 4.0
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_USE_BIAS = False
    TRAIN_DEVICE = "auto"
    TRAIN_EPOCHS = 1
    TRAIN_MAX_STEPS = 40000 #None
    TRAIN_LR = 2e-4
    TRAIN_WEIGHT_DECAY = 0.1
    TRAIN_GRAD_CLIP = 1.0
    TRAIN_LOG_EVERY = 20
    TRAIN_SAVE_DIR = "reports/checkpoints"
    TRAIN_SAVE_EVERY = 2000 #500
    TRAIN_LOSS_WINDOW = 50
    TRAIN_PLATEAU_ENABLED = False
    TRAIN_PLATEAU_FACTOR = 0.5
    TRAIN_PLATEAU_PATIENCE = 5
    TRAIN_PLATEAU_THRESHOLD = 1e-4
    TRAIN_PLATEAU_MIN_LR = 1e-6
    TRAIN_PLATEAU_COOLDOWN = 1
    TRAIN_SCHEDULER = "cosine"
    TRAIN_WARMUP_STEPS = 500      # ~1,3% dari TRAIN_MAX_STEPS
    TRAIN_MIN_LR = 2e-5           # 10% dari TRAIN_LR
    TRAIN_EARLY_STOP_ENABLED = False
    TRAIN_EARLY_STOP_PATIENCE = 10
    TRAIN_EARLY_STOP_MIN_DELTA = 1e-4

    TRAIN_GRAD_ACCUM = 8

    ALPHA = 1.0
    BETA = 0.5
    GAMMA = 0.3
    MIN_PAIR_FREQ = 100 # Keep as is, good for filtering less useful merges
    MAX_TOKEN_LENGTH = 18
    MAX_MERGES = VOCAB_SIZE - 4 # Set to target vocab size minus special tokens
    PRUNE_EVERY = 200 # Keep as is
    MERGES_PER_ROUND = 32 # Keep as is
    LENGTH_BONUS = 0.2
    NORMALIZE_WHITESPACE = True
    # Memory optimization for large-scale training
    MAX_VOCAB_SAMPLES = 500000   # Process max 500K lines for initial vocab
    MAX_COUNTER_SIZE = 5000000   # Prune counter at 5M entries
    PAIR_SUBSAMPLE_RATIO = 0.3   # Use 30% of corpus for pair scoring

    DATASET_BATCH_SIZE = 16 # Increased for larger effective batch size (16 * 8 = 128)
    DATASET_SHUFFLE = True # Essential for good training
    DATASET_NUM_WORKERS = 4 # Increased for faster data loading
    DATASET_BLOCK_SIZE = 256 #128
    DATASET_TYPE = "packed"
    DATASET_PIN_MEMORY = False # Enabled for CUDA performance
    DATASET_PERSISTENT_WORKERS = False # Enabled for DataLoader performance
    DATASET_PREFETCH_SIZE = 2
    DATASET_USE_MULTIPROCESSING = False
    DATASET_DROP_LAST = True
    DATASET_FAIL_FAST = True
    SLIDING_STRIDE = 1
    SLIDING_STREAMING = True
    SLIDING_CACHE_SIZE = 64 # Increased cache size

    BASE_DIR = PROJECT_ROOT / "data" / "prod"
    CORPUS_FILE = PROJECT_ROOT / "data" / "corpus.txt"
    DATASET_TOKEN_FILE = BASE_DIR / "tokens_prod.bin"
    VOCAB_PATH = BASE_DIR / "vocab_prod.json"
    TOKEN2ID_PATH = BASE_DIR / "token2id_prod.json"
    ID2TOKEN_PATH = BASE_DIR / "id2token_prod.json"
    PRETOKENIZE_OUTPUT_PATH = DATASET_TOKEN_FILE


# ========================================
# Testing Configuration
# ========================================
class TestingConfig(Config):
    """Konfigurasi untuk testing"""
    VOCAB_SIZE = 100
    MIN_FREQ = 1
    MAX_LENGTH = 128
    EMBEDDING_DIM = 64
    EMBEDDING_DROPOUT = 0.0
    POSITIONAL_EMBEDDING_TYPE = "sinusoidal"
    ROPE_BASE = 10000.0
    EMBEDDING_MAX_SEQ_LEN = 128
    ATTENTION_NUM_HEADS = 4
    ATTENTION_DROPOUT = 0.0
    ATTENTION_USE_BIAS = False
    ATTENTION_MAX_SEQ_LEN = 128
    TRANSFORMER_NUM_LAYERS = 2
    TRANSFORMER_MLP_RATIO = 4.0
    TRANSFORMER_DROPOUT = 0.0
    TRANSFORMER_USE_BIAS = False
    TRAIN_DEVICE = "cpu"
    TRAIN_EPOCHS = 1
    TRAIN_MAX_STEPS = 20
    TRAIN_LR = 1e-3
    TRAIN_WEIGHT_DECAY = 0.0
    TRAIN_GRAD_CLIP = 1.0
    TRAIN_LOG_EVERY = 5
    TRAIN_SAVE_DIR = "reports/checkpoints"
    TRAIN_SAVE_EVERY = 0
    TRAIN_LOSS_WINDOW = 10
    TRAIN_PLATEAU_ENABLED = True
    TRAIN_PLATEAU_FACTOR = 0.5
    TRAIN_PLATEAU_PATIENCE = 2
    TRAIN_PLATEAU_THRESHOLD = 1e-4
    TRAIN_PLATEAU_MIN_LR = 1e-6
    TRAIN_PLATEAU_COOLDOWN = 0
    TRAIN_EARLY_STOP_ENABLED = True
    TRAIN_EARLY_STOP_PATIENCE = 4
    TRAIN_EARLY_STOP_MIN_DELTA = 1e-4
    TRAIN_GRAD_ACCUM = 1
    
    MIN_PAIR_FREQ = 1
    PRUNE_EVERY = 20
    MERGES_PER_ROUND = 6
    LENGTH_BONUS = 0.12
    DATASET_BATCH_SIZE = 8
    DATASET_SHUFFLE = False
    DATASET_NUM_WORKERS = 0
    DATASET_BLOCK_SIZE = 64
    DATASET_TYPE = "packed"
    DATASET_PIN_MEMORY = False
    DATASET_PERSISTENT_WORKERS = False
    DATASET_PREFETCH_SIZE = 0
    DATASET_USE_MULTIPROCESSING = False
    DATASET_FAIL_FAST = True
    SLIDING_STRIDE = 1
    SLIDING_STREAMING = False
    SLIDING_CACHE_SIZE = 4

    BASE_DIR = PROJECT_ROOT / "data" / "test"
    CORPUS_FILE = PROJECT_ROOT / "data" / "corpus.txt"
    DATASET_TOKEN_FILE = BASE_DIR / "tokens_test.bin"
    VOCAB_PATH = BASE_DIR / "vocab_test.json"
    TOKEN2ID_PATH = BASE_DIR / "token2id_test.json"
    ID2TOKEN_PATH = BASE_DIR / "id2token_test.json"
    PRETOKENIZE_OUTPUT_PATH = DATASET_TOKEN_FILE

    DEFAULT_CORPUS = [
        "test one",
        "test two",
    ]


# ========================================
# Config Factory
# ========================================
def get_config(env: str = "development") -> Config:
    """
    Get config berdasarkan environment

    Args:
        env: "development", "production", atau "testing"

    Returns:
        Config class
    """
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
    }

    if env not in configs:
        raise ValueError(f"Unknown environment: {env}. Choose from {list(configs.keys())}")

    return configs[env]


# ========================================
# Default Config (dapat diubah via environment variable)
# ========================================
ENVIRONMENT = os.getenv("TOKENIZER_ENV", "development")
CONFIG = get_config(ENVIRONMENT)


if __name__ == "__main__":
    print("=== Default Configuration (Config Base) ===")
    print(f"VOCAB_SIZE: {Config.VOCAB_SIZE}")
    print(f"MAX_SUB_LEN: {Config.MAX_SUB_LEN}")
    print(f"ALPHA: {Config.ALPHA}, BETA: {Config.BETA}, GAMMA: {Config.GAMMA}")
    print(f"MIN_PAIR_FREQ: {Config.MIN_PAIR_FREQ}, PRUNE_EVERY: {Config.PRUNE_EVERY}")
    print(
        f"PAD_TOKEN: {Config.PAD_TOKEN}, UNK_TOKEN: {Config.UNK_TOKEN}, "
        f"BOS_TOKEN: {Config.BOS_TOKEN}, EOS_TOKEN: {Config.EOS_TOKEN}"
    )

    print("\nFile Paths:")
    print(f"CORPUS_FILE: {Config.CORPUS_FILE}")
    print(f"VOCAB_PATH: {Config.VOCAB_PATH}")
    print(f"TOKEN2ID_PATH: {Config.TOKEN2ID_PATH}")
    print(f"ID2TOKEN_PATH: {Config.ID2TOKEN_PATH}")

    print("\nCorpus (from file or default):")
    corpus = Config.get_corpus()
    for i, text in enumerate(corpus, 1):
        print(f"  {i}. {text}")

    print("\n\n=== All Configurations ===")
    for env_name in ["development", "production", "testing"]:
        env_config = get_config(env_name)
        print(f"\n{env_name.upper()}:")
        print(env_config.to_dict())
