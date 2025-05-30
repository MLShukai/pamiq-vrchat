from typing import override

import torch
import torch.nn as nn
from torch import Tensor


class StackedHiddenState(nn.Module):
    """Stacked hidden state for a sequence of modules.

    This module takes a sequence of modules and applies them to the
    input tensor and the hidden state tensor. It stacks the hidden
    states from each module and returns the output tensor and the
    stacked hidden state tensor.
    """

    def __init__(self, module_list: nn.ModuleList):
        """Initialize the StackedHiddenState module.

        Args:
            module_list: A list of modules to apply to the input tensor and the hidden state tensor.
        """
        super().__init__()
        self.module_list = module_list

    @override
    def forward(self, x: Tensor, hidden_stack: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the stacked hidden state module.

        Args:
            x: The input tensor of shape (*batch, len, dim) or (len, dim) or (*batch, dim) or (dim).
            hidden_stack: The hidden state tensor of shape (*batch, depth, len, dim) or (depth, len, dim) or (*batch, depth, dim) or (depth, dim).
        Returns:
            The output tensor of shape (*batch, len, dim) or (len, dim) or (*batch, dim) or (dim).
            The stacked hidden state tensor of shape (*batch, depth, len, dim) or (depth, len, dim) or (*batch, depth, dim) or (depth, dim).
        """
        no_batch = len(hidden_stack.shape) < 3
        if no_batch:
            x = x.unsqueeze(0)
            hidden_stack = hidden_stack.unsqueeze(0)

        no_len = len(x.shape) < 3
        if no_len:
            x = x.unsqueeze(1)

        batch_shape = x.shape[:-2]
        x = x.reshape(-1, *x.shape[len(batch_shape) :])
        hidden_stack = hidden_stack.reshape(-1, *hidden_stack.shape[len(batch_shape) :])

        hidden_out_list = []
        for i, module in enumerate(self.module_list):
            x, hidden_out = module(x, hidden_stack[:, i, :])
            hidden_out_list.append(hidden_out)

        hidden_out_stack = torch.stack(hidden_out_list).transpose(1, 0)

        x = x.view(*batch_shape, *x.shape[1:])
        hidden_out_stack = hidden_out_stack.view(
            *batch_shape, *hidden_out_stack.shape[1:]
        )

        if no_len:
            x = x.squeeze(1)
            hidden_out_stack = hidden_out_stack.squeeze(2)

        if no_batch:
            x = x.squeeze(0)
            hidden_out_stack = hidden_out_stack.squeeze(0)

        return x, hidden_out_stack
