from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .stacked_hidden_state import StackedHiddenState


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """Initialize the RMSNorm layer.

        Args:
            dim: The number of features in the input.
            eps: A small value to avoid division by zero.
            elementwise_affine: If True, use learnable parameters for scaling.
        """
        super().__init__()
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight: nn.Parameter | None = (
            nn.Parameter(torch.ones(dim)) if self.elementwise_affine else None
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the RMSNorm layer.

        Args:
            x: The input tensor of shape (*, dim).
        Returns:
            The output tensor of shape (*, dim).
        """
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            output = output * self.weight
        return output


class FFNSwiGLU(nn.Module):
    """Feed Forward Network with SwiGLU activation."""

    def __init__(self, dim: int, dim_ff_hidden: int):
        """Initialize the FFNSwiGLU layer.

        Args:
            dim: The number of features in the input.
            dim_ff_hidden: The number of features in the hidden layer.
        """
        super().__init__()
        self.fc = nn.Linear(dim, dim_ff_hidden)
        self.fc_act = nn.Linear(dim, dim_ff_hidden)
        self.fc_out = nn.Linear(dim_ff_hidden, dim)
        self.act = nn.SiLU()

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed forward network with SwiGLU activation.

        Args:
            x: The input tensor of shape (*, dim).
        Returns:
            The output tensor of shape (*, dim).
        """
        x = self.fc(x) * self.act(self.fc_act(x))
        x = self.fc_out(x)
        return x


def scan(a: Tensor, b: Tensor) -> Tensor:
    """Calculate a sequence [b0, a1 * b0 + b1, a2 * a1 * b0 + a2 * b1, b2, ...]
    from [a0, a1, a2, ...] and [b0, b1, b2, ...]

    Args:
        a: (batch, len)
        b: (batch, len)
    Returns:
        a sequence defined above, of shape (batch, len)
    """
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
    """QLSTM Layer, which is a variant of LSTM without h recursion.

    It uses a parallel scan to compute the hidden state. The QLSTM layer
    is designed to be used in a QLSTM block.
    """

    def __init__(self, dim: int):
        """Initialize the QLSTM layer.

        Args:
            dim: The number of features in the input.
        """
        super().__init__()
        self.dim = dim
        self.fc_forget = nn.Linear(dim, dim)
        self.fc_input = nn.Linear(dim, dim)
        self.fc_input_gate = nn.Linear(dim, dim)
        self.fc_output_gate = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    @override
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the QLSTM layer.

        Args:
            x: The input tensor of shape (batch, len, dim).
            hidden: The hidden state tensor of shape (batch, dim).
        Returns:
            The output tensor of shape (batch, len, dim) and the new hidden state tensor of shape (batch, len, dim).
        """
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
    """QLSTM Block, which consists of a QLSTM layer and a feed forward
    network."""

    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float):
        """Initialize the QLSTM block.

        Args:
            dim: The number of features in the input.
            dim_ff_hidden: The number of features in the hidden layer.
            dropout: The dropout rate.
        """
        super().__init__()
        self.qlstm = QLSTMLayer(dim)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_qlstm = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the QLSTM block.

        Args:
            x: The input tensor of shape (batch, len, dim).
            hidden: The hidden state tensor of shape (batch, len, dim).
        Returns:
            The output tensor of shape (batch, len, dim) and the new hidden state tensor of shape (batch, len, dim).
        """
        x_ = x
        x = self.norm_qlstm(x)
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
    """QLSTM, which is a stack of QLSTM blocks."""

    def __init__(self, depth: int, dim: int, dim_ff_hidden: int, dropout: float):
        """Initialize the QLSTM.

        Args:
            depth: The number of QLSTM blocks.
            dim: The number of features in the input.
            dim_ff_hidden: The number of features in the hidden layer.
            dropout: The dropout rate.
        """
        super().__init__(
            nn.ModuleList(
                [QLSTMBlock(dim, dim_ff_hidden, dropout) for _ in range(depth)]
            )
        )
