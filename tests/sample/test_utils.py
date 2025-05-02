import pytest

from sample.utils import size_2d, size_2d_to_int_tuple


@pytest.mark.parametrize("input,expected", [(10, (10, 10)), ((2, 3), (2, 3))])
def test_size_2d_to_int_tuple(input: size_2d, expected: tuple[int, int]):
    assert size_2d_to_int_tuple(input) == expected
