from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import CONFIG
from embedding import build_tokenizer_and_embedding, resolve_embedding_max_seq_len


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        self.dim = int(dim)
        self.base = float(base)
        self.max_seq_len_cached = 0
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._set_cos_sin_cache(max_seq_len, device=inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype: torch.dtype):
        seq_len = int(seq_len)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype=dtype)
        self.sin_cached = emb.sin().to(dtype=dtype)
        self.max_seq_len_cached = seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.size(-1) // 2]
        x2 = x[..., x.size(-1) // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(position_offset) + q.size(-2)
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != q.device:
            self._set_cos_sin_cache(seq_len, device=q.device, dtype=torch.float32)

        cos = self.cos_cached[position_offset:seq_len].to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached[position_offset:seq_len].to(device=q.device, dtype=q.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with optional Grouped Query Attention (GQA).

    GQA (Grouped Query Attention):
    - Digunakan oleh: LLaMA 2 70B, LLaMA 3, Mistral, Gemma, DeepSeek
    - n_kv_head < n_head → lebih efisien di memory dan inference
    - n_kv_head = n_head → standard MHA (backward compatible)

    Arsitektur:
        Q: n_head heads (query)
        K: n_kv_head heads (key)  → di-repeat untuk match Q
        V: n_kv_head heads (value) → di-repeat untuk match Q

    Contoh:
        n_head=8, n_kv_head=4 → setiap KV head melayani 2 query head
        n_head=8, n_kv_head=8 → standard MHA (GQA disabled)
        n_head=8, n_kv_head=2 → GQA agresif (hemat memory besar)
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 2048,
        positional_embedding_type: str = "learned",
        rope_base: float = 10000.0,
        num_layers: int = 12,
    ):
        super().__init__()

        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        assert self.n_head % self.n_kv_head == 0, (
            f"n_head ({n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
        )
        self.num_query_per_kv = self.n_head // self.n_kv_head
        self.head_dim = d_model // n_head
        self.num_layers = int(num_layers)
        self.max_seq_len = int(max_seq_len)
        self.scale = self.head_dim ** -0.5
        self.mask_fill_value = -1e4
        self.positional_embedding_type = str(positional_embedding_type)

        self.rotary_emb = (
            RotaryEmbedding(self.head_dim, max_seq_len=self.max_seq_len, base=rope_base)
            if self.positional_embedding_type == "rope"
            else None
        )

        # GQA: Q projection (n_head) terpisah dari KV projection (n_kv_head)
        q_dim = self.n_head * self.head_dim          # = d_model
        kv_dim = self.n_kv_head * self.head_dim      # < d_model jika GQA aktif

        self.q_proj = nn.Linear(d_model, q_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj.is_residual_projection = True

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)

        std = 0.02
        if getattr(self.out_proj, "is_residual_projection", False):
            std = std / math.sqrt(2 * self.num_layers)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _mask_fill_value_for_dtype(self, dtype: torch.dtype) -> float:
        if dtype.is_floating_point:
            return torch.finfo(dtype).min
        return self.mask_fill_value

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Expand KV heads untuk match query heads.

        Input:  (B, n_kv_head, T, head_dim)
        Output: (B, n_head, T, head_dim)  where n_head = n_kv_head * n_rep

        Args:
            x: tensor KV (B, n_kv_head, T, head_dim)
            n_rep: jumlah repeat per KV head (num_query_per_kv)

        Returns:
            Tensor (B, n_head, T, head_dim)
        """
        if n_rep == 1:
            return x

        B, n_kv, T, D = x.shape
        # (B, n_kv, 1, T, D) → expand → (B, n_kv, n_rep, T, D) → reshape → (B, n_kv * n_rep, T, D)
        x = x[:, :, None, :, :]
        x = x.expand(B, n_kv, n_rep, T, D)
        return x.reshape(B, n_kv * n_rep, T, D)

    def _compute_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V projections.

        Output shapes:
            Q: (B, n_head, T, head_dim)
            K: (B, n_kv_head, T, head_dim)
            V: (B, n_kv_head, T, head_dim)
        """
        B, T, _ = x.size()

        q = self.q_proj(x)  # (B, T, n_head * head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_head * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_head * head_dim)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        return q, k, v

    def _make_causal_allow_mask(self, q_len: int, k_len: int, past_len: int, device) -> torch.Tensor:
        query_positions = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(-1)
        key_positions = torch.arange(k_len, device=device).unsqueeze(0)
        return key_positions <= query_positions  # (q_len, k_len)

    def _resolve_key_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        key_len: int,
        device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        if attention_mask.dim() == 2:
            key_mask = attention_mask.to(device=device)
            if key_mask.size(0) != batch_size:
                raise ValueError(
                    f"attention_mask batch mismatch: got {key_mask.size(0)}, expected {batch_size}."
                )
            if key_mask.size(1) < key_len:
                raise ValueError(
                    f"attention_mask length {key_mask.size(1)} is smaller than key length {key_len}."
                )
            if key_mask.size(1) > key_len:
                key_mask = key_mask[:, -key_len:]
            return key_mask != 0

        if attention_mask.dim() == 4:
            return None

        raise ValueError(
            "attention_mask must be 2D (B, T) or 4D broadcastable tensor (B, 1, Q, K)."
        )

    def _normalize_4d_attention_mask(
        self,
        attention_mask: torch.Tensor,
        q: torch.Tensor,
        q_len: int,
        k_len: int,
        past_len: int,
    ) -> torch.Tensor:
        mask = attention_mask.to(device=q.device)
        expected_shape = (q.size(0), self.n_head, q_len, k_len)
        try:
            torch.broadcast_shapes(tuple(mask.shape), expected_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"4D attention_mask shape {tuple(mask.shape)} is not broadcastable to {expected_shape}."
            ) from exc

        causal_allow = self._make_causal_allow_mask(
            q_len=q_len,
            k_len=k_len,
            past_len=past_len,
            device=q.device,
        ).unsqueeze(0).unsqueeze(0)

        if mask.dtype == torch.bool or not mask.dtype.is_floating_point:
            allow = mask if mask.dtype == torch.bool else mask != 0
            return allow & causal_allow

        causal_mask = torch.zeros((1, 1, q_len, k_len), device=q.device, dtype=q.dtype)
        causal_mask = causal_mask.masked_fill(
            ~causal_allow,
            self._mask_fill_value_for_dtype(q.dtype),
        )
        return mask.to(dtype=q.dtype) + causal_mask

    def _build_attn_mask(
        self,
        q: torch.Tensor,
        q_len: int,
        k_len: int,
        past_len: int,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], bool, Optional[torch.Tensor]]:
        B = q.size(0)
        device = q.device

        if attention_mask is not None and attention_mask.dim() == 4:
            attn_mask = self._normalize_4d_attention_mask(
                attention_mask=attention_mask,
                q=q,
                q_len=q_len,
                k_len=k_len,
                past_len=past_len,
            )
            return attn_mask, False, None

        key_mask = self._resolve_key_mask(
            attention_mask=attention_mask,
            batch_size=B,
            key_len=k_len,
            device=device,
        )

        if past_len == 0 and key_mask is None:
            # Fast path: let backend generate causal mask (flash/mem-efficient kernels).
            return None, True, None

        causal_allow = self._make_causal_allow_mask(q_len=q_len, k_len=k_len, past_len=past_len, device=device)
        allow = causal_allow.unsqueeze(0).unsqueeze(0).expand(B, 1, q_len, k_len)

        if key_mask is not None:
            allow = allow & key_mask[:, None, None, :]

        attn_mask = torch.zeros((B, 1, q_len, k_len), device=device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~allow, self._mask_fill_value_for_dtype(q.dtype))

        query_mask = None
        if key_mask is not None:
            query_mask = key_mask[:, -q_len:]

        return attn_mask, False, query_mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """
        x: (B, T, D)

        attention_mask: optional
          - (B, T) with 1 for valid tokens, 0 for padded tokens
          - or 4D broadcastable to (B, n_head, Q, K).

        past_key_value: optional tuple (k, v)
          - Shapes: (B, n_kv_head, T_cache, head_dim)
          - Untuk GQA, KV cache menyimpan n_kv_head, BUKAN n_head.
            Ini yang membuat GQA lebih hemat memory dibanding MHA.

        use_cache:
          - False: return output tensor (B, T, D)
          - True: return tuple (output, (k, v))
        """

        B, T, D = x.size()
        if D != self.d_model:
            raise ValueError(f"Input hidden dim {D} does not match d_model={self.d_model}.")
        if T > self.max_seq_len and past_key_value is None:
            raise ValueError(
                f"Input sequence length {T} exceeds max_seq_len={self.max_seq_len}. "
                "Increase ATTENTION_MAX_SEQ_LEN / EMBEDDING_MAX_SEQ_LEN in config."
            )

        q, k, v = self._compute_qkv(x)
        # q: (B, n_head, T, head_dim)
        # k: (B, n_kv_head, T, head_dim)
        # v: (B, n_kv_head, T, head_dim)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.dim() != 4 or past_v.dim() != 4:
                raise ValueError("past_key_value tensors must be rank-4: (B, n_head, T, head_dim).")
            if past_k.shape[:2] != (B, self.n_kv_head) or past_v.shape[:2] != (B, self.n_kv_head):
                raise ValueError(
                    f"past_key_value head mismatch. "
                    f"Expected (B={B}, n_kv_head={self.n_kv_head}), "
                    f"got past_k={past_k.shape[:2]}, past_v={past_v.shape[:2]}"
                )
            if past_k.shape[-1] != self.head_dim or past_v.shape[-1] != self.head_dim:
                raise ValueError("past_key_value head_dim does not match current attention head_dim.")
            past_len = past_k.size(-2)

        # Apply RoPE BEFORE repeat_kv (lebih efisien)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, position_offset=past_len)

        # Concatenate with cache (masih dalam format n_kv_head)
        if past_key_value is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # GQA: expand KV heads untuk match Q heads
        k = self.repeat_kv(k, self.num_query_per_kv)  # (B, n_head, T+K, head_dim)
        v = self.repeat_kv(v, self.num_query_per_kv)  # (B, n_head, T+K, head_dim)

        # Pastikan tidak melebihi max_seq_len
        if k.size(-2) > self.max_seq_len:
            k = k[:, :, -self.max_seq_len:, :]
            v = v[:, :, -self.max_seq_len:, :]
            past_len = max(0, k.size(-2) - T)

        q_len = q.size(-2)
        k_len = k.size(-2)

        attn_mask, is_causal, query_mask = self._build_attn_mask(
            q=q,
            q_len=q_len,
            k_len=k_len,
            past_len=past_len,
            attention_mask=attention_mask,
        )

        try:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )
        except TypeError:
            # Fallback untuk torch < 2.1 (tanpa scale argument)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
        # (B, n_head, T, head_dim)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, D)

        # Final projection
        output = self.out_proj(attn_output)

        if query_mask is not None:
            output = output * query_mask[:, :, None].to(dtype=output.dtype)

        if use_cache:
            # KV cache dalam format n_kv_head (lebih hemat memory)
            # Ambil k, v SEBELUM repeat_kv agar cache tetap kecil
            # Tapi kita sudah concat dengan cache, jadi kita perlu
            # k, v yang sudah di-concat tapi BELUM di-repeat.
            # Solusi: simpan k, v dalam bentuk n_kv_head.
            # Karena k/v sudah di-concat di atas, kita bisa simpan langsung.
            # Tapi masalah: kita sudah concat, lalu repeat. Kita perlu yang sebelum repeat.
            # Mari kita reconstruct dari k yang sudah di-repeat:
            # Ambil sample pertama per group

            if self.num_query_per_kv > 1:
                # k sudah di-repeat: (B, n_head, T_total, head_dim)
                # Kita perlu extract kembali ke n_kv_head dengan mengambil
                # setiap num_query_per_kv-th head
                present_k = k[:, :: self.num_query_per_kv, :, :].contiguous()
                present_v = v[:, :: self.num_query_per_kv, :, :].contiguous()
            else:
                present_k = k.contiguous()
                present_v = v.contiguous()
            present = (present_k, present_v)
            return output, present

        return output

    @classmethod
    def from_config(
        cls,
        cfg=CONFIG,
        d_model: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ):
        resolved_d_model = int(d_model if d_model is not None else getattr(cfg, "EMBEDDING_DIM", 256))
        resolved_n_head = int(getattr(cfg, "ATTENTION_NUM_HEADS", 8))
        resolved_n_kv_head = int(getattr(cfg, "ATTENTION_NUM_KV_HEADS", 0))

        if resolved_n_kv_head <= 0:
            resolved_n_kv_head = resolved_n_head  # MHA mode (backward compatible)

        if resolved_d_model % resolved_n_head != 0:
            raise ValueError(
                f"EMBEDDING_DIM ({resolved_d_model}) must be divisible by ATTENTION_NUM_HEADS ({resolved_n_head})."
            )
        if resolved_n_head % resolved_n_kv_head != 0:
            raise ValueError(
                f"ATTENTION_NUM_HEADS ({resolved_n_head}) must be divisible by "
                f"ATTENTION_NUM_KV_HEADS ({resolved_n_kv_head})."
            )

        resolved_max_seq_len = resolve_attention_max_seq_len(cfg=cfg, override=max_seq_len)
        return cls(
            d_model=resolved_d_model,
            n_head=resolved_n_head,
            n_kv_head=resolved_n_kv_head,
            dropout=float(getattr(cfg, "ATTENTION_DROPOUT", 0.1)),
            bias=bool(getattr(cfg, "ATTENTION_USE_BIAS", False)),
            max_seq_len=resolved_max_seq_len,
            positional_embedding_type=str(getattr(cfg, "POSITIONAL_EMBEDDING_TYPE", "learned")),
            rope_base=float(getattr(cfg, "ROPE_BASE", 10000.0)),
            num_layers=int(getattr(cfg, "TRANSFORMER_NUM_LAYERS", 12)),
        )

    def forward_batch(
        self,
        batch: dict,
        embedding_layer: nn.Module,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_offset: Optional[int] = None,
    ):
        """
        End-to-end helper from dataset batch: input_ids -> embedding -> attention output.
        """
        if "input_ids" not in batch:
            raise KeyError("Batch must contain input_ids.")

        if position_offset is None:
            position_offset = 0 if past_key_value is None else int(past_key_value[0].size(-2))

        hidden = embedding_layer.forward_batch(batch, position_offset=position_offset)
        attention_mask = batch.get("attention_mask")
        return self.forward(
            hidden,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )


def resolve_attention_max_seq_len(cfg=CONFIG, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)

    configured = getattr(cfg, "ATTENTION_MAX_SEQ_LEN", None)
    if configured is not None:
        return int(configured)

    # Fallback to embedding sequence limit, then dataset block size.
    return resolve_embedding_max_seq_len(cfg=cfg, override=None)


def build_attention_from_config(
    cfg=CONFIG,
    d_model: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> MultiHeadSelfAttention:
    return MultiHeadSelfAttention.from_config(
        cfg=cfg,
        d_model=d_model,
        max_seq_len=max_seq_len,
    )


def build_tokenizer_embedding_attention(cfg=CONFIG, load_tokenizer: bool = True):
    tokenizer, embedding = build_tokenizer_and_embedding(cfg=cfg, load_tokenizer=load_tokenizer)
    attention = build_attention_from_config(
        cfg=cfg,
        d_model=embedding.token_embedding.embedding_dim,
        max_seq_len=embedding.max_seq_len,
    )
    return tokenizer, embedding, attention

