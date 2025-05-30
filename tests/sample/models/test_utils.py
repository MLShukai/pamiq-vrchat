import pytest
import torch
import torch.nn as nn

from sample.models.utils import init_weights


class TestInitWeights:
    def test_linear_weight_initialization(self):
        """Test Linear layer weight initialization."""
        # Create a linear layer
        linear = nn.Linear(10, 5)
        init_std = 0.02

        # Apply initialization
        init_weights(linear, init_std)

        # Check weight distribution properties
        # For truncated normal, we can check if standard deviation is close to init_std
        # (not exactly equal due to truncation)
        assert 0.01 < torch.std(linear.weight).item() < 0.03

        # Check bias is initialized to zeros
        assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))

    def test_conv2d_initialization(self):
        """Test Conv2d layer initialization."""
        # Create a conv layer
        conv = nn.Conv2d(3, 6, kernel_size=3)
        init_std = 0.02

        # Apply initialization
        init_weights(conv, init_std)

        # Check weight distribution properties
        assert 0.01 < torch.std(conv.weight).item() < 0.03

        # Check bias is initialized to zeros
        assert conv.bias is not None
        assert torch.allclose(conv.bias, torch.zeros_like(conv.bias))

    def test_conv_transpose_2d_initialization(self):
        """Test Conv2d layer initialization."""
        # Create a conv layer
        conv = nn.ConvTranspose2d(3, 6, kernel_size=3)
        init_std = 0.02

        # Apply initialization
        init_weights(conv, init_std)

        # Check weight distribution properties
        assert 0.01 < torch.std(conv.weight).item() < 0.03

        # Check bias is initialized to zeros
        assert conv.bias is not None
        assert torch.allclose(conv.bias, torch.zeros_like(conv.bias))

    def test_layernorm_initialization(self):
        """Test LayerNorm initialization."""
        # Create a layer norm
        norm = nn.LayerNorm(10)
        init_std = 0.02

        # Apply initialization
        init_weights(norm, init_std)

        # Check weight is initialized to ones
        assert torch.allclose(norm.weight, torch.ones_like(norm.weight))

        # Check bias is initialized to zeros
        assert torch.allclose(norm.bias, torch.zeros_like(norm.bias))
