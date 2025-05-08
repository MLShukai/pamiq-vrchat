import pytest
import torch
from torch.distributions import Categorical
from torch.distributions.distribution import Distribution

from sample.models.components.multi_discretes import (
    FCMultiCategoricalHead,
    MultiCategoricals,
    MultiEmbeddings,
)


class TestMultiCategoricals:
    @pytest.fixture
    def distributions(self) -> list[Categorical]:
        choices_per_dist = [3, 2, 5]
        batch_size = 8
        return [
            Categorical(logits=torch.zeros(batch_size, c)) for c in choices_per_dist
        ]

    @pytest.fixture
    def multi_categoricals(self, distributions) -> MultiCategoricals:
        return MultiCategoricals(distributions)

    def test_init(self, multi_categoricals: MultiCategoricals):
        assert multi_categoricals.batch_shape == (8, 3)

    def test_sample(self, multi_categoricals: MultiCategoricals):
        assert multi_categoricals.sample().shape == (8, 3)
        assert multi_categoricals.sample((1, 2)).shape == (1, 2, 8, 3)

    def test_log_prob(self, multi_categoricals: MultiCategoricals):
        sampled = multi_categoricals.sample()
        assert multi_categoricals.log_prob(sampled).shape == sampled.shape

    def test_entropy(self, multi_categoricals: MultiCategoricals):
        assert multi_categoricals.entropy().shape == (8, 3)


class TestFCMultiCategoricalHead:
    @pytest.mark.parametrize(
        """
        batch,
        dim_in,
        choices_per_category,
        """,
        [
            (8, 256, [3, 3, 3, 2, 2]),
            (1, 16, [1, 2, 3, 4, 5]),
        ],
    )
    def test_fc_multi_categorical_head(self, batch, dim_in, choices_per_category):
        policy = FCMultiCategoricalHead(dim_in, choices_per_category)
        input = torch.randn(batch, dim_in)
        dist = policy(input)
        assert isinstance(dist, Distribution)
        assert dist.sample().shape == (batch, len(choices_per_category))
        assert dist.log_prob(dist.sample()).shape == (batch, len(choices_per_category))
        assert dist.entropy().shape == (batch, len(choices_per_category))


class TestMultiEmbeddings:
    @pytest.mark.parametrize(
        """
        choices_per_category,
        embedding_dim,
        """,
        [
            ([3, 4, 5], 128),
            ([5, 4, 5, 7], 123),
        ],
    )
    def test_choices_per_category(self, choices_per_category, embedding_dim):
        """Test that choices_per_category property returns the correct
        values."""
        me = MultiEmbeddings(choices_per_category, embedding_dim)
        assert me.choices_per_category == choices_per_category

    @pytest.mark.parametrize(
        """
        batch,
        length,
        choices_per_category,
        embedding_dim,
        """,
        [
            (32, 64, [3, 4, 5], 128),
            (3, 5, [5, 4, 5, 7], 123),
        ],
    )
    def test_no_flatten(self, batch, length, choices_per_category, embedding_dim):
        """Test output shape when do_flatten is False."""
        me = MultiEmbeddings(choices_per_category, embedding_dim)
        input_list = []
        for choices in choices_per_category:
            input_list.append(torch.randint(choices, (batch, length)))
        input = torch.stack(input_list, dim=-1)
        output = me(input)
        assert output.shape == (batch, length, len(choices_per_category), embedding_dim)

    @pytest.mark.parametrize(
        """
        batch,
        length,
        choices_per_category,
        embedding_dim,
        """,
        [
            (32, 64, [3, 4, 5], 128),
            (3, 5, [5, 4, 5, 7], 123),
        ],
    )
    def test_do_flatten(self, batch, length, choices_per_category, embedding_dim):
        """Test output shape when do_flatten is True."""
        me = MultiEmbeddings(choices_per_category, embedding_dim, do_flatten=True)
        input_list = []
        for choices in choices_per_category:
            input_list.append(torch.randint(choices, (batch, length)))
        input = torch.stack(input_list, dim=-1)
        output = me(input)
        assert output.shape == (
            batch,
            length,
            len(choices_per_category) * embedding_dim,
        )
