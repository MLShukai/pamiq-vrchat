import pytest
import torch
from torch.distributions import Normal

from sample.models.components.deterministic_normal import (
    SCALE_ONE,
    SHIFT_ZERO,
    DeterministicNormal,
    FCDeterministicNormalHead,
)


def test_logprob_shiftzero():
    """Test that the shift (minimum value) of negative log-likelihood equals
    the squared error.

    The scale differs by a factor of `math.pi`.
    """

    mean = torch.zeros(10)
    std = torch.full_like(mean, SHIFT_ZERO)
    normal = Normal(mean, std)
    expected = torch.zeros_like(mean)

    torch.testing.assert_close(normal.log_prob(mean), expected)


def test_logprob_scaleone():
    """Test that the scale of negative log-likelihood equals the squared error.

    The error shift differs by `0.5 * math.log(math.pi)`.
    """

    mean = torch.zeros(3)
    std = torch.full_like(mean, SCALE_ONE)
    normal = Normal(mean, std)

    t1 = torch.full_like(mean, 1)
    t2 = torch.full_like(mean, 3)

    nlp1 = -normal.log_prob(t1)
    nlp2 = -normal.log_prob(t2)

    expected = (t1 - mean) ** 2 - (t2 - mean) ** 2
    actual = nlp1 - nlp2

    torch.testing.assert_close(actual, expected)


class TestFCDeterministicNormalHead:
    def test_forward(self):
        """Test the forward pass returns a Normal distribution with expected
        shape."""
        layer = FCDeterministicNormalHead(10, 20)
        out = layer(torch.randn(10))

        assert isinstance(out, DeterministicNormal)
        assert out.sample().shape == (20,)

        assert layer(torch.randn(1, 2, 3, 10)).sample().shape == (1, 2, 3, 20)

    def test_squeeze_feature_dim(self):
        """Test the squeeze_feature_dim parameter works correctly."""
        with pytest.raises(ValueError):
            # out_features must be 1 when squeeze_feature_dim=True
            FCDeterministicNormalHead(10, 2, squeeze_feature_dim=True)

        # squeeze_feature_dim default is False
        FCDeterministicNormalHead(10, 2)

        net = FCDeterministicNormalHead(10, 1, squeeze_feature_dim=True)
        x = torch.randn(10)
        out = net(x)
        assert out.sample().shape == ()


class TestDeterministicNormal:
    def test_sample(self):
        """Test that sample and rsample always return the mean."""
        mean = torch.randn(10)
        std = torch.ones(10)
        dn = DeterministicNormal(mean, std)

        # All sample calls should return the mean exactly
        assert torch.equal(dn.sample(), mean)
        assert torch.equal(dn.sample(), mean)
        assert torch.equal(dn.sample(), mean)

        # All rsample calls should also return the mean exactly
        assert torch.equal(dn.rsample(), mean)
        assert torch.equal(dn.rsample(), mean)
        assert torch.equal(dn.rsample(), mean)
