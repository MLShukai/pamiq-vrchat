from typing import cast, override

import torch
import torch.nn as nn
from torch import Tensor


class MultiEmbeddings(nn.Module):
    """Convert multi-discrete inputs to embedding vectors.

    This module creates separate embedding layers for each discrete
    category and processes multi-dimensional discrete inputs by
    embedding each category independently. It's useful for handling
    complex discrete state spaces or multi-discrete action spaces in
    reinforcement learning.
    """

    def __init__(
        self,
        choices_per_category: list[int],
        embedding_dim: int,
        do_flatten: bool = False,
    ) -> None:
        """Initialize the multi-embedding module.

        Args:
            choices_per_category: A list of choice sizes for each category.
            embedding_dim: The dimension of each embedding vector.
            do_flatten: If True, flatten the output embeddings across categories.
        """
        super().__init__()

        self.do_flatten = do_flatten
        self.embeds = nn.ModuleList()
        for choice in choices_per_category:
            self.embeds.append(nn.Embedding(choice, embedding_dim))

    @property
    def choices_per_category(self) -> list[int]:
        """Get the number of choices for each category.

        Returns:
            A list containing the number of possible values for each category.
        """

        return [e.num_embeddings for e in cast(list[nn.Embedding], self.embeds)]

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Convert multi-discrete inputs to embedding vectors.

        Args:
            input: Tensor of discrete indices with shape (*, num_categories),
                where num_categories equals len(choices_per_category).

        Returns:
            Embedded tensor with shape:
            - (*, num_categories * embedding_dim) if do_flatten is True
            - (*, num_categories, embedding_dim) if do_flatten is False
        """
        batch_list = []
        for layer, tensor in zip(self.embeds, input.movedim(-1, 0)):
            batch_list.append(layer(tensor))

        output = torch.stack(batch_list, dim=-2)
        if self.do_flatten:
            output = output.reshape(
                *output.shape[:-2], output.shape[-2] * output.shape[-1]
            )
        return output
