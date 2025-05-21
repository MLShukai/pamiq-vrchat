import pytest
import torch

from sample.utils import average_exponentially, size_2d, size_2d_to_int_tuple
from tests.sample.helpers import parametrize_device


@pytest.mark.parametrize("input,expected", [(10, (10, 10)), ((2, 3), (2, 3))])
def test_size_2d_to_int_tuple(input: size_2d, expected: tuple[int, int]):
    assert size_2d_to_int_tuple(input) == expected


class TestAverageExponentially:
    @pytest.mark.parametrize("shape", [(4,), (2, 3, 4), (10, 1, 1)])
    def test_output_shape(self, shape):
        out = average_exponentially(torch.randn(shape))
        assert out.shape == shape[1:]

    def test_invalid_input(self):
        with pytest.raises(
            ValueError, match="Input sequence dimension must be larger than 1d!"
        ):
            average_exponentially(torch.randn(()))

    @parametrize_device
    def test_device_compatibility(self, device):
        """Test function works on different devices."""
        sequence = torch.tensor([1.0, 2.0, 3.0], device=device)
        result = average_exponentially(sequence)

        assert result.device == device
        assert result.dtype == sequence.dtype
