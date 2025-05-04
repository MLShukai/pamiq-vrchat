# SioConv with Parallel Scan  (PS)
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .stacked_hidden_state import StackedHiddenState


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight: nn.Parameter | None = (
            nn.Parameter(torch.ones(dim)) if self.elementwise_affine else None
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            output = output * self.weight
        return output


class FFNSwiGLU(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim_ff_hidden)
        self.fc_act = nn.Linear(dim, dim_ff_hidden)
        self.fc_out = nn.Linear(dim_ff_hidden, dim)
        self.act = nn.SiLU()

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x) * self.act(self.fc_act(x))
        x = self.fc_out(x)
        return x


# [a0, a1, a2, ...], [b0, b1, b2, ...] -> [b0, a1 * b0 + b1, a2 * a1 * b0 + a2 * b1, b2, ...]
def scan(a, b):
    _, length = a.shape
    if length == 1:
        return b
    is_odd = length % 2 == 1
    a_even = a[:, : -1 if is_odd else None : 2]
    a_odd = a[:, 1::2]
    b_even = b[:, : -1 if is_odd else None : 2]
    b_odd = b[:, 1::2]
    mask_odd = torch.zeros(length, device=a.device, dtype=a.dtype)
    mask_odd[1::2] = 1
    mask_odd = mask_odd[None, :]
    b_new = torch.addcmul(
        torch.addcmul(b, b, mask_odd, value=-1),
        F.pad(
            scan(a_odd * a_even, torch.addcmul(b_odd, a_odd, b_even)).repeat_interleave(
                2, dim=1
            ),
            (0, 1) if is_odd else (0, 0),
            value=0,
        ),
        mask_odd,
    )
    b_odd_new = b_new[:, 1 : None if is_odd else -1 : 2]
    a_even_new = a[:, 2::2]
    mask_even = torch.zeros(length, device=a.device, dtype=a.dtype)
    mask_even[2::2] = 1
    mask_even = mask_even[None, :]
    b_new = torch.addcmul(
        b_new,
        F.pad(
            (a_even_new * b_odd_new).repeat_interleave(2, dim=1),
            (1, 0) if is_odd else (1, 1),
            value=0,
        ),
        mask_even,
    )
    return b_new


class QLSTMLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc_forget = nn.Linear(dim, dim)
        self.fc_input = nn.Linear(dim, dim)
        self.fc_input_gate = nn.Linear(dim, dim)
        self.fc_output_gate = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.last_hidden = None
        self.last_hidden_init = nn.Parameter(torch.randn(dim))
        self.is_refresh = True

    # (batch, len, dim), (batch, dim) -> (batch, len, dim), (batch, len, dim)
    @override
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch, len, dim = x.shape

        forget = F.sigmoid(self.fc_forget(x))  # (batch, len, dim)

        input = self.tanh(self.fc_input(x)) * self.sigmoid(self.fc_input_gate(x))
        h_inner_chunk = (
            scan(
                forget.transpose(2, 1).reshape(batch * dim, len),
                input.transpose(2, 1).reshape(batch * dim, len),
            )
            .reshape(batch, dim, len)
            .transpose(2, 1)
        )

        h = torch.addcmul(h_inner_chunk, hidden[:, None, :], forget.cumprod(1))

        y = self.tanh(h) * self.sigmoid(self.fc_output_gate(x))
        return y, h


class QLSTMBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float):
        super().__init__()
        self.qlstm = QLSTMLayer(dim)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_sioconv = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        x_ = x
        x = self.norm_sioconv(x)
        x, hidden = self.qlstm(x, hidden)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x, hidden


class QLSTM(StackedHiddenState):
    def __init__(self, depth: int, dim: int, dim_ff_hidden: int, dropout: float):
        super().__init__(
            nn.ModuleList(
                [QLSTMBlock(dim, dim_ff_hidden, dropout) for _ in range(depth)]
            )
        )
