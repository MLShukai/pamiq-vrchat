import pytest
import torch

from sample.transforms.common import Standardize


class TestStandardize:
    @pytest.mark.parametrize(
        "input_shape",
        [(100,), (3, 32, 32), (5, 10, 10), (2, 3, 64, 64)],
    )
    def test_standardization(self, input_shape):
        transform = Standardize()
        input_tensor = torch.randn(input_shape) * 5 + 3
        output = transform(input_tensor)

        assert output.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert output.std().item() == pytest.approx(1.0, abs=1e-6)

    def test_constant_tensor(self):
        transform = Standardize(eps=1e-8)
        input_tensor = torch.ones(10, 10) * 5.0
        output = transform(input_tensor)

        assert torch.allclose(output, torch.zeros_like(output))

    def test_single_value_tensor(self):
        transform = Standardize()
        input_tensor = torch.tensor([42.0])
        output = transform(input_tensor)

        assert output.shape == (1,)
        assert output.item() == pytest.approx(0.0)

    def test_empty_tensor(self):
        transform = Standardize()
        input_tensor = torch.tensor([])
        output = transform(input_tensor)

        assert output.shape == (0,)
