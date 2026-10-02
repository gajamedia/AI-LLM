from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import CONFIG
from dataset2 import create_dataloader
from embedding import EmbeddingLayer, build_tokenizer_and_embedding
from transformerStack import TransformerStack, resolve_transformer_max_seq_len
from modules.rmsnorm import RMSNorm


class GPTModel(nn.Module):
    """
    Full GPT-style decoder-only language model.

    Architecture:
        Embedding
        → TransformerStack
        → LM Head (weight tied)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 2048,
        tie_weights: bool = True,
        embedding_dropout: Optional[float] = None,
        positional_embedding_type: str = "learned",
        rope_base: float = 10000.0,
        embedding_layer: Optional[EmbeddingLayer] = None,
        mlp_type: str = "swiglu",
        norm_type: str = "rmsnorm",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_kv_head = n_kv_head
        self.tie_weights = tie_weights

        # Embedding (token + positional)
        if embedding_layer is not None:
            embedding_vocab_size, embedding_d_model = embedding_layer.token_embedding.weight.shape
            if embedding_vocab_size != vocab_size:
                raise ValueError(
                    "embedding_layer vocab size does not match model vocab_size. "
                    f"embedding={embedding_vocab_size}, model={vocab_size}."
                )
            if embedding_d_model != d_model:
                raise ValueError(
                    "embedding_layer dimension does not match model d_model. "
                    f"embedding={embedding_d_model}, model={d_model}."
                )

        self.embedding = embedding_layer or EmbeddingLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=float(dropout if embedding_dropout is None else embedding_dropout),
            pos_type=positional_embedding_type,
        )
        self.token_embedding = self.embedding.token_embedding

        # Transformer stack
        self.transformer = TransformerStack(
            num_layers=num_layers,
            d_model=d_model,
            n_head=n_head,
            n_kv_head=self.n_kv_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            bias=bias,
            max_seq_len=max_seq_len,
            positional_embedding_type=positional_embedding_type,
            rope_base=rope_base,
            final_norm=False,
            mlp_type=mlp_type,
            norm_type=norm_type,
        )

        # Final Normalization
        if str(norm_type).lower() == "layernorm":
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = RMSNorm(d_model)

        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize seluruh module
        self.apply(self._init_weights)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # self._init_weights()

    # ============================================================
    # Initialization
    # ============================================================

    # def _init_weights(self):
    #     if not self.tie_weights:
    #         nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            std = 0.02

            if getattr(module, "is_residual_projection", False):
                std /= math.sqrt(2 * self.transformer.num_layers)

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=std,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # RMSNorm
        elif hasattr(module, "weight") and module.__class__.__name__ == "RMSNorm":
            nn.init.ones_(module.weight)

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        position_offset: Optional[int] = None,
        document_ids: Optional[torch.Tensor] = None,
    ):
        """
        input_ids: (B, T)
        labels: (B, T) optional for training
        position_offset: starting position for positional embeddings.
            If omitted with past_key_values, defaults to cached key length.

        Returns:
            if labels:
                loss, logits
            else:
                logits
            optionally returns cache
        """

        if document_ids is not None:
            if document_ids.shape != input_ids.shape:
                raise ValueError("document_ids must have the same shape as input_ids.")
            segment_ids = document_ids.to(device=input_ids.device)
            valid_segments = segment_ids >= 0
            segment_mask = (
                (segment_ids[:, :, None] == segment_ids[:, None, :])
                & valid_segments[:, :, None]
                & valid_segments[:, None, :]
            ).unsqueeze(1)
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    existing_mask = (attention_mask != 0)[:, None, None, :]
                elif attention_mask.ndim == 4:
                    existing_mask = attention_mask.to(dtype=torch.bool)
                else:
                    raise ValueError("attention_mask must be 2D or 4D when document_ids are provided.")
                segment_mask = segment_mask & existing_mask
            attention_mask = segment_mask

        # Embedding (token + positional)
        if position_offset is None:
            position_offset = (
                0
                if past_key_values is None
                else int(past_key_values[0][0].size(-2))
            )
        x = self.embedding(input_ids, position_offset=position_offset)

        # Transformer
        if use_cache:
            hidden_states, new_past = self.transformer(
                x,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            hidden_states = self.transformer(
                x,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=False,
            )
            new_past = None

        # Final Norm
        hidden_states = self.norm(hidden_states)

        # LM Head
        logits = self.lm_head(hidden_states)

        # Training loss
        if labels is not None:
            # Hitung CE dalam fp32 agar log-sum-exp stabil terhadap logits bf16/fp16.
            # logits yang dikembalikan tetap dalam dtype model untuk menghemat memori.
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )

            if use_cache:
                return loss, logits, new_past
            return loss, logits

        if use_cache:
            return logits, new_past

        return logits

    def load_compatible_state_dict(self, state_dict, strict: bool = True):
        """Load current checkpoints and discard the former stack-level norm keys."""
        current_keys = set(self.state_dict())
        cleaned = dict(state_dict)
        legacy_norm_keys = {
            key for key in cleaned
            if key.startswith("transformer.final_ln.") and key not in current_keys
        }
        for key in legacy_norm_keys:
            cleaned.pop(key)
        return self.load_state_dict(cleaned, strict=strict)

    def forward_batch(
        self,
        batch: Dict[str, torch.Tensor],
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_offset: Optional[int] = None,
    ):
        if "input_ids" not in batch:
            raise KeyError("Batch must contain 'input_ids'.")
        return self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=batch.get("labels"),
            position_offset=position_offset,
            document_ids=batch.get("document_ids"),
        )

    # ============================================================
    # Factory from config
    # ============================================================

    @classmethod
    def from_config(
        cls,
        cfg=CONFIG,
        vocab_size: Optional[int] = None,
        embedding_layer: Optional[EmbeddingLayer] = None,
    ):
        resolved_vocab_size = int(
            vocab_size if vocab_size is not None else getattr(cfg, "VOCAB_SIZE", 20000)
        )
        d_model = int(getattr(cfg, "EMBEDDING_DIM", 768))
        n_head = int(getattr(cfg, "ATTENTION_NUM_HEADS", 12))
        n_kv_head = int(getattr(cfg, "ATTENTION_NUM_KV_HEADS", 0))
        if n_kv_head <= 0:
            n_kv_head = n_head  # MHA mode
        num_layers = int(
            getattr(cfg, "TRANSFORMER_NUM_LAYERS", getattr(cfg, "NUM_LAYERS", 12))
        )

        if d_model % n_head != 0:
            raise ValueError(
                f"EMBEDDING_DIM ({d_model}) must be divisible by "
                f"ATTENTION_NUM_HEADS ({n_head})."
            )
        if n_head % n_kv_head != 0:
            raise ValueError(
                f"ATTENTION_NUM_HEADS ({n_head}) must be divisible by "
                f"ATTENTION_NUM_KV_HEADS ({n_kv_head})."
            )

        return cls(
            vocab_size=resolved_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            n_head=n_head,
            n_kv_head=n_kv_head,
            mlp_ratio=float(
                getattr(cfg, "TRANSFORMER_MLP_RATIO", getattr(cfg, "MLP_RATIO", 4.0))
            ),
            dropout=float(
                getattr(cfg, "TRANSFORMER_DROPOUT", getattr(cfg, "DROPOUT", 0.1))
            ),
            attention_dropout=float(getattr(cfg, "ATTENTION_DROPOUT", 0.1)),
            bias=bool(
                getattr(
                    cfg,
                    "TRANSFORMER_USE_BIAS",
                    getattr(cfg, "USE_BIAS", getattr(cfg, "ATTENTION_USE_BIAS", False)),
                )
            ),
            max_seq_len=resolve_transformer_max_seq_len(cfg=cfg, override=None),
            tie_weights=bool(getattr(cfg, "TIE_WORD_EMBEDDINGS", True)),
            embedding_dropout=float(
                getattr(cfg, "EMBEDDING_DROPOUT", getattr(cfg, "DROPOUT", 0.1))
            ),
            positional_embedding_type=str(
                getattr(cfg, "POSITIONAL_EMBEDDING_TYPE", "learned")
            ),
            rope_base=float(getattr(cfg, "ROPE_BASE", 10000.0)),
            embedding_layer=embedding_layer,
            mlp_type=str(getattr(cfg, "MLP_TYPE", "swiglu")),
            norm_type=str(getattr(cfg, "NORMALIZATION_TYPE", "rmsnorm")),
        )

    @classmethod
    def from_config_with_tokenizer(cls, cfg=CONFIG, load_tokenizer: bool = True):
        tokenizer, embedding = build_tokenizer_and_embedding(
            cfg=cfg,
            load_tokenizer=load_tokenizer,
        )
        model = cls.from_config(
            cfg=cfg,
            vocab_size=len(tokenizer.token2id),
            embedding_layer=embedding,
        )
        return model, tokenizer

    @classmethod
    def build_training_components(
        cls,
        cfg=CONFIG,
        load_tokenizer: bool = True,
        token_file: Optional[str] = None,
        **dataloader_kwargs,
    ):
        model, tokenizer = cls.from_config_with_tokenizer(
            cfg=cfg,
            load_tokenizer=load_tokenizer,
        )
        # Build dataloader defaults from the same cfg to avoid env mismatch
        # between model/tokenizer and dataset pipeline.
        dataloader_params = {
            "token_file": token_file or str(getattr(cfg, "DATASET_TOKEN_FILE")),
            "block_size": int(getattr(cfg, "DATASET_BLOCK_SIZE", 128)),
            "batch_size": int(getattr(cfg, "DATASET_BATCH_SIZE", 16)),
            "shuffle": bool(getattr(cfg, "DATASET_SHUFFLE", True)),
            "num_workers": int(getattr(cfg, "DATASET_NUM_WORKERS", 0)),
            "pin_memory": bool(getattr(cfg, "DATASET_PIN_MEMORY", True)),
            "persistent_workers": bool(getattr(cfg, "DATASET_PERSISTENT_WORKERS", False)),
            "drop_last": bool(getattr(cfg, "DATASET_DROP_LAST", False)),
            "dataset_type": str(getattr(cfg, "DATASET_TYPE", "packed")),
            "stride": int(getattr(cfg, "SLIDING_STRIDE", 1)),
            "dtype": getattr(cfg, "DATASET_DTYPE", None),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": int(tokenizer.token2id[tokenizer.pad_token]),
        }
        dataloader_params.update(dataloader_kwargs)
        dataloader = create_dataloader(**dataloader_params)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "dataloader": dataloader,
        }
