import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Norm dihitung dalam fp32 supaya mean/rsqrt tidak melimpah di fp16/bf16,
        # lalu hasilnya dikembalikan ke dtype input.
        orig_dtype = x.dtype
        x_float = x.float() if orig_dtype in (torch.float16, torch.bfloat16) else x
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_float = x_float * torch.rsqrt(rms + self.eps)
        return (self.weight * x_float).to(orig_dtype)