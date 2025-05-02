# Ref: https://github.com/facebookresearch/ijepa

import numpy as np
import numpy.typing as npt

from sample.utils import size_2d, size_2d_to_int_tuple


def get_2d_positional_embeddings(
    embed_dim: int, grid_size: size_2d
) -> npt.NDArray[np.float64]:
    """
    Args:
        embed_dim: dim of positional embeddings.
        grid_size: int of the grid height and width.
    Returns:
        positional embeddings (shape: [grid_size_h, grid_size_w, embed_dim]).
    """
    grid_size_h, grid_size_w = size_2d_to_int_tuple(grid_size)
    grid_h = np.arange(grid_size_h, dtype=float)
    grid_w = np.arange(grid_size_w, dtype=float)
    meshgrid = np.meshgrid(grid_w, grid_h)  # here w goes first as args
    grid = np.stack(meshgrid, axis=0)  # [2, grid_size_h, grid_size_w]

    positional_embeddings = _get_2d_sincos_positional_embeddings_from_grid(
        embed_dim, grid
    )
    return positional_embeddings


def _get_2d_sincos_positional_embeddings_from_grid(
    embed_dim: int, grid: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Args:
        embed_dim: dim of positional embeddings.
        grid: positions to be encoded is represented as grid(shape: [2, grid_size_h, grid_size_w]).
    Returns:
        positional embeddings (shape: [grid_size_h, grid_size_w, embed_dim]).
    """

    assert embed_dim % 2 == 0
    assert grid.shape[0] == 2  # grid_h, grid_w

    # use half of dimensions to encode grid_h
    embeddings_h = _get_1d_sincos_positional_embeddings(
        embed_dim // 2, grid[0].reshape(-1)
    )  # [grid_size_h*grid_size_w, embed_dim//2]
    embeddings_w = _get_1d_sincos_positional_embeddings(
        embed_dim // 2, grid[1].reshape(-1)
    )  # [grid_size_h*grid_size_w, embed_dim//2]

    embeddings = np.concatenate(
        [embeddings_h, embeddings_w], axis=-1
    )  # [grid_size_h*grid_size_w, embed_dim]
    _, grid_size_h, grid_size_w = grid.shape
    return embeddings.reshape(grid_size_h, grid_size_w, embed_dim)


def _get_1d_sincos_positional_embeddings(
    embed_dim: int, positions: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Args:
        embed_dim: dim of positional embeddings.
        positions: positions to be encoded (shape: [length, ]).
    Returns:
        positional embeddings (shape: [length, embed_dim]).
    """
    assert embed_dim % 2 == 0
    assert positions.ndim == 1
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # [embed_dim//2]
    outer = np.outer(positions, omega)  # [length, embed_dim//2]

    positional_embeddings_sin = np.sin(outer)  # [length, embed_dim//2]
    positional_embeddings_cos = np.cos(outer)  # [length, embed_dim//2]

    positional_embeddings = np.concatenate(
        [positional_embeddings_sin, positional_embeddings_cos], axis=-1
    )  # [length, embed_dim]
    return positional_embeddings
