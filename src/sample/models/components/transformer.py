# Ref: https://github.com/facebookresearch/ijepa

import math
from functools import partial
from typing import override

import torch
import torch.nn as nn

from ..utils import init_weights


class MLP(nn.Module):
    """Multi Layer Perceptron.

    A simple feed-forward neural network with GELU activation and
    dropout.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the MLP module.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features.
            out_features: Number of output features.
            dropout: Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the MLP.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """Attention Layer.

    Multi-head self-attention mechanism as used in Transformers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether to add bias to the QKV projection.
            qk_scale: Scaling factor for attention weights. If None, uses head_dim ** -0.5.
            attn_drop: Attention dropout probability.
            proj_drop: Output projection dropout probability.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if qk_scale is None:
            qk_scale = head_dim**-0.5
        assert qk_scale != 0.0
        self.scale = qk_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Attention module.

        Args:
            x: Input tensor of shape [batch_size, num_patches, embedding_dim].

        Returns:
            A tuple containing:
                - Output tensor of shape [batch_size, num_patches, embedding_dim]
                - Attention weights of shape [batch_size, num_heads, num_patches, num_patches]
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class TransformerLayer(nn.Module):
    """Transformer Layer.

    A standard Transformer layer consisting of multi-head self-attention
    followed by a feed-forward MLP, with residual connections and layer
    normalization.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        """Initialize the TransformerLayer.

        Args:
            embedding_dim: Dimension of the embeddings.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dimension to embedding dimension.
            qkv_bias: Whether to add bias to the QKV projection.
            qk_scale: Scaling factor for dot products in attention.
            dropout: Dropout probability for MLP and projection.
            attn_drop: Dropout probability for attention weights.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.attn = Attention(
            embedding_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embedding_dim,
            hidden_features=mlp_hidden_dim,
            out_features=embedding_dim,
            dropout=dropout,
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Transformer.

        Args:
            x: Shape is [batch_size, n_patches, embedding_dim]

        Returns:
            Shape is same as input.
        """
        y, _ = self.attn(self.norm1(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


def _fix_init_weight(layer: TransformerLayer, depth: int) -> None:
    """Rescale the weights of projection layers in transformer blocks by depth.

    This rescaling improves training stability for deeper networks by adjusting
    the variance of the output distribution based on network depth.

    Args:
        layer: TransformerLayer to rescale weights.
        depth: Current depth of this layer in the network (starting from 1).

    Raises:
        ValueError: If depth is zero.
    """
    if depth == 0:
        raise ValueError("Depth must be non-zero.")

    def rescale(param: torch.Tensor) -> None:
        param.div_(math.sqrt(2.0 * depth))

    # layer: TransformerLayer
    rescale(layer.attn.proj.weight.data)
    rescale(layer.mlp.fc2.weight.data)


class Transformer(nn.Module):
    """Transformer Encoder.

    A stack of Transformer layers that processes sequences of token
    embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        init_std: float = 0.02,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """Initialize the Transformer.

        Args:
            embedding_dim: Dimension of the embeddings.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dimension to embedding dimension.
            qkv_bias: Whether to add bias to the QKV projection.
            qk_scale: Scaling factor for dot products in attention.
            dropout: Dropout probability for MLP and projection.
            attn_drop: Dropout probability for attention weights.
            init_std: Standard deviation for weight initialization.
            norm_layer: Normalization layer class.
        """
        super().__init__()

        # Build transformer layers
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = TransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout=dropout,
                attn_drop=attn_drop,
            )
            layer.apply(partial(init_weights, init_std=init_std))
            _fix_init_weight(layer, i + 1)
            self.blocks.append(layer)

        # Final normalization layer
        self.norm = norm_layer(embedding_dim)
        init_weights(self.norm, init_std)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input embeddings through the transformer layers.

        Args:
            x: Input tensor of shape [batch_size, seq_length, embedding_dim].

        Returns:
            Output tensor of shape [batch_size, seq_length, embedding_dim].
        """
        # Apply each transformer block in sequence
        for block in self.blocks:
            x = block(x)

        # Apply final normalization
        x = self.norm(x)

        return x
