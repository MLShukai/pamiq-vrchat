from typing import override

import pytest
import torch
import torch.nn as nn

from sample.models.components.transformer import (
    MLP,
    Attention,
    Transformer,
    TransformerLayer,
)


class TestMLP:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 32])
    @pytest.mark.parametrize("in_features", [128, 256])
    @pytest.mark.parametrize("hidden_features", [256, 512])
    @pytest.mark.parametrize("out_features", [128, 256])
    def test_mlp_shape(
        self, batch_size, seq_len, in_features, hidden_features, out_features
    ):
        """Test MLP output shape is correct."""
        mlp = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            dropout=0.0,
        )

        x = torch.randn(batch_size, seq_len, in_features)
        out = mlp(x)

        assert out.shape == (batch_size, seq_len, out_features)

    def test_mlp_dropout_zero(self):
        """Test MLP with dropout=0 gives deterministic outputs."""
        mlp = MLP(in_features=128, hidden_features=256, out_features=128, dropout=0.0)

        x = torch.randn(2, 16, 128)
        out1 = mlp(x)
        out2 = mlp(x)

        # With dropout=0, outputs should be identical
        assert torch.allclose(out1, out2)

    def test_mlp_dropout_effect(self):
        """Test MLP applies dropout during training."""
        torch.manual_seed(42)  # For reproducibility

        mlp = MLP(
            in_features=128,
            hidden_features=256,
            out_features=128,
            dropout=0.5,  # High dropout for testing
        )

        # Set to training mode
        mlp.train()

        x = torch.randn(2, 16, 128)
        out1 = mlp(x)
        out2 = mlp(x)

        # With high dropout and training mode, outputs should differ
        assert not torch.allclose(out1, out2)

        # In eval mode, even with dropout, outputs should be deterministic
        mlp.eval()
        out3 = mlp(x)
        out4 = mlp(x)
        assert torch.allclose(out3, out4)


class TestAttention:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 32])
    @pytest.mark.parametrize("dim", [128, 256])
    @pytest.mark.parametrize("num_heads", [4, 8])
    def test_attention_shape(self, batch_size, seq_len, dim, num_heads):
        """Test Attention output shapes are correct."""
        attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
        )

        x = torch.randn(batch_size, seq_len, dim)
        out, attn_weights = attn(x)

        assert out.shape == (batch_size, seq_len, dim)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_attention_dropout_effect(self):
        """Test attention dropout affects outputs during training."""
        torch.manual_seed(42)  # For reproducibility

        attn = Attention(
            dim=128,
            num_heads=4,
            attn_drop=0.5,  # High dropout for testing
            proj_drop=0.5,
        )

        # Set to training mode
        attn.train()

        x = torch.randn(2, 16, 128)
        out1, _ = attn(x)
        out2, _ = attn(x)

        # With high dropout and training mode, outputs should differ
        assert not torch.allclose(out1, out2)

        # In eval mode, outputs should be deterministic
        attn.eval()
        out3, _ = attn(x)
        out4, _ = attn(x)
        assert torch.allclose(out3, out4)


class TestTransformerLayer:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 32])
    @pytest.mark.parametrize("embedding_dim", [128, 256])
    @pytest.mark.parametrize("num_heads", [4, 8])
    def test_transformer_layer_shape(
        self, batch_size, seq_len, embedding_dim, num_heads
    ):
        """Test TransformerLayer output shape matches input shape."""
        layer = TransformerLayer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            dropout=0.0,
            attn_drop=0.0,
        )

        x = torch.randn(batch_size, seq_len, embedding_dim)
        out = layer(x)

        assert out.shape == x.shape


class TestTransformer:
    """Test the Transformer class."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [16, 32])
    @pytest.mark.parametrize("embedding_dim", [128, 256])
    @pytest.mark.parametrize("depth", [1, 4])
    @pytest.mark.parametrize("num_heads", [4, 8])
    def test_transformer_shape(
        self, batch_size, seq_len, embedding_dim, depth, num_heads
    ):
        """Test Transformer output shape matches input shape."""
        transformer = Transformer(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=0.0,
            attn_drop=0.0,
        )

        x = torch.randn(batch_size, seq_len, embedding_dim)
        out = transformer(x)

        assert out.shape == x.shape

    def test_transformer_layer_count(self):
        """Test Transformer creates the correct number of layers."""
        depth = 6
        embedding_dim = 128
        num_heads = 4

        transformer = Transformer(
            embedding_dim=embedding_dim, depth=depth, num_heads=num_heads
        )

        assert len(transformer.blocks) == depth
        for block in transformer.blocks:
            assert isinstance(block, TransformerLayer)

    def test_custom_norm_layer(self):
        """Test custom normalization layer works."""
        embedding_dim = 128
        depth = 2
        num_heads = 4

        # Create a custom norm layer
        class CustomNorm(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm = nn.LayerNorm(dim)

            @override
            def forward(self, x):
                return self.norm(x)

        transformer = Transformer(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            norm_layer=CustomNorm,
        )

        assert isinstance(transformer.norm, CustomNorm)
