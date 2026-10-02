from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed Forward Network

    Digunakan pada:
        - LLaMA
        - LLaMA 2
        - LLaMA 3
        - Mistral
        - Mixtral
        - Gemma
        - DeepSeek
        - Qwen

    Arsitektur:

        gate = SiLU(Wg x)
        up   = Wu x

        hidden = gate * up

        output = Wd hidden
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Gate Projection
        self.gate_proj = nn.Linear(
            d_model,
            hidden_dim,
            bias=bias,
        )

        # Value Projection
        self.up_proj = nn.Linear(
            d_model,
            hidden_dim,
            bias=bias,
        )

        # Output Projection
        self.down_proj = nn.Linear(
            hidden_dim,
            d_model,
            bias=bias,
        )

        # Tandai sebagai residual projection.
        # Akan digunakan GPTModel._init_weights()
        self.down_proj.is_residual_projection = True

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate = F.silu(self.gate_proj(x))

        value = self.up_proj(x)

        x = gate * value

        x = self.down_proj(x)

        x = self.dropout(x)

        return x

    @staticmethod
    def compute_swiglu_hidden_dim(
        d_model: int,
        multiple_of: int = 256,
        multiplier: Optional[float] = None,
    ) -> int:
        """
        Hidden dimension yang dipakai LLaMA.

        Default:
            hidden = ceil((8/3 * d_model) / multiple_of) * multiple_of

        multiplier dapat dipakai bila ingin
        mengubah ukuran FFN.
        """

        hidden = int((8 * d_model) / 3)

        if multiplier is not None:
            hidden = int(hidden * multiplier)

        hidden = multiple_of * (
            (hidden + multiple_of - 1)
            // multiple_of
        )

        return hidden