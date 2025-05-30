from collections.abc import Iterable
from typing import cast, override

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution, constraints


class MultiCategoricals(Distribution):
    """Set of separate Categorical distributions representing multiple discrete
    action spaces.

    This distribution handles multiple independent categorical
    distributions, each potentially having different numbers of
    categories. It's useful for environments with multiple discrete
    action dimensions, like game controllers with different buttons or
    modes.
    """

    @property
    @override
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Returns empty constraints."""
        return {}

    def __init__(self, categoricals: Iterable[Categorical]) -> None:
        """Constructs Multi Categorical distribution from a collection of
        Categorical distributions.

        Args:
            categoricals: A collection of Categorical distributions, where each distribution may
                have a different number of categories but must share the same batch shape.

        Raises:
            ValueError: If the collection is empty or if the batch shapes don't match.
        """

        categoricals = list(categoricals)
        if len(categoricals) == 0:
            raise ValueError("Input categoricals collection is empty.")

        first_dist = categoricals[0]

        if not all(first_dist.batch_shape == d.batch_shape for d in categoricals):
            raise ValueError("All batch shapes must be same.")

        batch_shape = torch.Size((*first_dist.batch_shape, len(categoricals)))
        super().__init__(
            batch_shape=batch_shape, event_shape=torch.Size(), validate_args=False
        )

        self.dists = categoricals

    @override
    def sample(self, sample_shape: Iterable[int] = torch.Size()) -> torch.Tensor:
        """Sample from each distribution and stack the outputs.

        Args:
            sample_shape: Shape of the samples to draw.

        Returns:
            Tensor of sampled actions with shape (*sample_shape, num_dists).
        """
        return torch.stack([d.sample(list(sample_shape)) for d in self.dists], dim=-1)

    @override
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions for each distribution.

        Args:
            value: Tensor of actions with shape (*, num_dists).

        Returns:
            Tensor of log probabilities with shape (*, num_dists).
        """
        return torch.stack(
            [d.log_prob(v) for d, v in zip(self.dists, value.movedim(-1, 0))], dim=-1
        )

    @override
    def entropy(self) -> torch.Tensor:
        """Compute entropy for each distribution.

        Returns:
            Tensor of entropies with shape (*, num_dists), where * is the batch shape.
        """
        return torch.stack([d.entropy() for d in self.dists], dim=-1)


class FCMultiCategoricalHead(nn.Module):
    """Fully connected multi-categorical distribution head.

    This module applies multiple linear transformations to the input
    features and returns a MultiCategoricals distribution. It's useful
    for producing policies over multiple discrete action spaces, such as
    in environments with compound discrete actions.
    """

    def __init__(self, dim_in: int, choices_per_category: list[int]) -> None:
        """Initialize the multi-categorical distribution head.

        Args:
            dim_in: Input dimension size of tensor.
            choices_per_category: List of category counts for each discrete action space.
        """
        super().__init__()

        self.heads = nn.ModuleList()
        for choice in choices_per_category:
            self.heads.append(nn.Linear(dim_in, choice, bias=False))

    @override
    def forward(self, input: torch.Tensor) -> MultiCategoricals:
        """Compute the multi-categorical distribution from input features.

        Args:
            input: Input tensor with shape (..., dim_in).

        Returns:
            A MultiCategoricals distribution representing multiple independent
            categorical distributions.
        """
        categoricals = []
        for head in self.heads:
            logits = head(input)
            categoricals.append(Categorical(logits=logits))

        return MultiCategoricals(categoricals)


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
    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
