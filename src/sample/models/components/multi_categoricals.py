from collections.abc import Iterable
from typing import override

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution


class MultiCategoricals(Distribution):
    """Set of separate Categorical distributions representing multiple discrete
    action spaces.

    This distribution handles multiple independent categorical
    distributions, each potentially having different numbers of
    categories. It's useful for environments with multiple discrete
    action dimensions, like game controllers with different buttons or
    modes.
    """

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
