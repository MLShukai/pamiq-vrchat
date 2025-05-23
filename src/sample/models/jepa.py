"""JEPA model components.

This module provides the components for the Joint Embedding Predictive
Architecture (JEPA) model.
"""

import copy
from collections.abc import Callable
from typing import Literal, Self, override

import torch
import torch.nn as nn
from pamiq_core.torch import get_device

from .components.transformer import Transformer
from .utils import init_weights


class Encoder(nn.Module):
    """Encoder for Joint Embedding Predictive Architecture (JEPA) with mask
    support."""

    def __init__(
        self,
        patchfier: Callable[[torch.Tensor], torch.Tensor],
        positional_encodings: torch.Tensor,
        hidden_dim: int = 768,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """Initialize the JEPAEncoder.

        Args:
            patchfier: Patchfy input data to patch sequence.
            positional_encodings: Positional encoding tensors to be added to patchfied input data.
            hidden_dim: Hidden dimension per patch.
            embed_dim: Output dimension per patch.
            depth: Number of transformer layers.
            num_heads: Number of attention heads for transformer layers.
            mlp_ratio: Ratio for MLP hidden dimension in transformer layers.
            qkv_bias: Whether to use bias in query, key, value projections.
            qk_scale: Scale factor for query-key dot product.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()
        self.num_features = self.embed_dim = hidden_dim
        self.num_heads = num_heads
        if positional_encodings.ndim != 2:
            raise ValueError("positional_encodings must be 2d tensor!")
        if positional_encodings.size(1) != hidden_dim:
            raise ValueError(
                "positional_encodings channel dimension must be hidden_dim."
            )

        self.patchfier = patchfier

        # define mask token_vector
        self.mask_token_vector = nn.Parameter(torch.empty(hidden_dim))

        self.register_buffer("positional_encodings", None)
        self.positional_encodings = positional_encodings.unsqueeze(0)

        # define transformer
        self.transformer = Transformer(
            embedding_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout=drop_rate,
            attn_drop=attn_drop_rate,
            init_std=init_std,
        )

        self.out_proj = nn.Linear(hidden_dim, embed_dim)

        # initialize
        nn.init.trunc_normal_(self.mask_token_vector, std=init_std)
        init_weights(self.out_proj, init_std)

    @override
    def forward(
        self, data: torch.Tensor, masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode input data into latents, applying masks if provided.

        Args:
            data: Input data
            masks: Boolean masks for images embedded as patches with shape
                [batch_size, n_patches]. True values indicate masked patches.

        Returns:
            Encoded latents with shape [batch_size, n_patches, out_dim]
        """
        # Patchify input data
        x = self.patchfier(data)
        # x: [batch_size, n_patches, embed_dim]

        # Apply mask if provided
        if masks is not None:
            if x.shape[:-1] != masks.shape:
                raise ValueError(
                    f"Shape mismatch: x{x.shape[:-1]} vs masks{masks.shape}"
                )
            if masks.dtype != torch.bool:
                raise ValueError(
                    f"Mask tensor dtype must be bool. input: {masks.dtype}"
                )
            x = x.clone()  # Avoid breaking gradient graph
            x[masks] = self.mask_token_vector

        # Add positional embedding to x
        x = x + self.positional_encodings

        # Apply transformer
        x = self.transformer(x)

        # Project to output dimension
        x = self.out_proj(x)
        return x

    @override
    def __call__(
        self, data: torch.Tensor, masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        return super().__call__(data, masks)

    def clone(self) -> Self:
        """Clone model for creating target or context encoder."""
        return copy.deepcopy(self)


class Predictor(nn.Module):
    """Predictor for Joint Embedding Predictive Architecture (JEPA) with target
    support."""

    def __init__(
        self,
        positional_encodings: torch.Tensor,
        embed_dim: int = 384,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """Initialize the JEPAPredictor.

        Args:
            positional_encodings: Positional encoding tensors to be added to patchfied input data.
                Shape is [num_patch, hidden_dim]
            hidden_dim: Hidden dimension for prediction.
            depth: Number of transformer layers.
            num_heads: Number of attention heads for transformer layers.
            mlp_ratio: Ratio for MLP hidden dimension in transformer layers.
            qkv_bias: Whether to use bias in query, key, value projections.
            qk_scale: Scale factor for query-key dot product.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()
        if positional_encodings.ndim != 2:
            raise ValueError("positional_encodings must be 2d tensor!")
        if positional_encodings.size(1) != hidden_dim:
            raise ValueError(
                "positional_encodings channel dimension must be hidden_dim."
            )

        self.input_proj = nn.Linear(embed_dim, hidden_dim, bias=True)

        # prepare tokens representing patches to be predicted
        self.prediction_token_vector = nn.Parameter(torch.empty(hidden_dim))

        # define positional encodings
        self.register_buffer("positional_encodings", None)
        self.positional_encodings = positional_encodings.unsqueeze(0)

        # define transformer
        self.transformer = Transformer(
            embedding_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout=drop_rate,
            attn_drop=attn_drop_rate,
            init_std=init_std,
        )

        self.predictor_proj = nn.Linear(hidden_dim, embed_dim, bias=True)

        # initialize
        nn.init.trunc_normal_(self.prediction_token_vector, std=init_std)
        init_weights(self.input_proj, init_std)
        init_weights(self.predictor_proj, init_std)

    @override
    def forward(
        self,
        latents: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Predict latents of target patches based on input latents and boolean
        targets.

        Args:
            latents: Input latents from context_encoder with shape
                [batch, n_patches, embed_dim]
            targets: Boolean targets for patches with shape [batch, n_patches].
                True values indicate target patches to be predicted.

        Returns:
            Prediction results for target patches with shape
                [batch, n_patches, embed_dim]
        """
        # Map from encoder-dim to predictor-dim
        x = self.input_proj(latents)

        # Apply targets: adding prediction tokens
        if x.shape[:-1] != targets.shape:
            raise ValueError(
                f"Shape mismatch: x{x.shape[:-1]} vs targets{targets.shape}"
            )
        if targets.dtype != torch.bool:
            raise ValueError(
                f"Target tensor dtype must be bool. input: {targets.dtype}"
            )

        x = x.clone()  # Avoid breaking gradient graph
        x[targets] += self.prediction_token_vector

        # Add positional encodings
        x = x + self.positional_encodings

        # Apply transformer
        x = self.transformer(x)

        # Project to output dimension
        x = self.predictor_proj(x)

        return x

    @override
    def __call__(self, latents: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return super().__call__(latents, targets)


class AveragePoolInfer:
    """Applies average pooling to encoded 1d (audio) or 2d (image) patches from
    a JEPA encoder."""

    def __init__(
        self,
        ndim: Literal[1, 2],
        num_patches: int | tuple[int, ...],
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
    ) -> None:
        """Initialize the average pooling inference wrapper.

        Args:
            ndim: Number of spatial dimensions (1 or 2 are supported).
            num_patches: Number of patches in the original encoded representation.
                If int, assumes uniform patches across all dimensions.
                If tuple, length must match ndim.
            kernel_size: Size of the pooling kernel.
                If int, uses same size for all dimensions.
                If tuple, length must match ndim.
            stride: Stride of the pooling operation.
                If None, defaults to kernel_size.
                If int, uses same stride for all dimensions.
                If tuple, length must match ndim.

        Raises:
            ValueError:
                - If num_patches, kernel_size, stride tuple length doesn't match ndim.
        """

        self.ndim = ndim

        def validate_and_normalize(obj: int | tuple[int, ...]) -> tuple[int, ...]:
            if isinstance(obj, int):
                return (obj,) * ndim
            if len(obj) != ndim:
                raise ValueError(f"Expected tuple of length {ndim}, got {len(obj)}")
            return obj

        # Validate and normalize num_patches
        self.num_patches = validate_and_normalize(num_patches)
        kernel_size = validate_and_normalize(kernel_size)
        if stride is not None:
            stride = validate_and_normalize(stride)

        # Create appropriate pooling layer
        match ndim:
            case 1:
                self.pool = nn.AvgPool1d(kernel_size, stride)  # pyright: ignore[reportArgumentType, ]
            case 2:
                self.pool = nn.AvgPool2d(kernel_size, stride)  # pyright: ignore[reportArgumentType, ]

    def __call__(self, encoder: Encoder, data: torch.Tensor) -> torch.Tensor:
        """Process data through the encoder and apply average pooling to the
        result.

        Args:
            encoder: JEPA Encoder instance.
            data: N-dimensional tensor with shape [*batch, dim, *spatial_dims].

        Returns:
            Tensor with shape [*batch, patch', dim] where patch' is the reduced number
            of patches after pooling.
        """
        device = get_device(encoder)
        data = data.to(device)

        # Handle batch dimension
        data_ndim = 1 + self.ndim  # dim + spatial dimensions
        if no_batch := data.ndim < data_ndim:
            data = data.unsqueeze(0)

        batch_shape = data.shape[:-data_ndim]
        data = data.reshape(-1, *data.shape[-data_ndim:])

        # Encode
        x = encoder(data)  # [batch', patch, dim]
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))

        x = x.transpose(-1, -2)  # [batch', dim, patch]

        # Reshape for pooling: [batch', dim, *spatial_patch_dims]
        x = x.reshape(-1, x.size(-2), *self.num_patches)

        # Apply pooling
        x: torch.Tensor = self.pool(x)

        # Flatten spatial dimensions and transpose back
        x = x.flatten(-self.ndim).transpose(-1, -2)  # [batch', patch', dim]

        # Restore original batch shape
        x = x.reshape(*batch_shape, *x.shape[-2:])
        if no_batch:
            x = x.squeeze(0)
        return x
