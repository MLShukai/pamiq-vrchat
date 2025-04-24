from typing import override

import pytest

from pamiq_vrchat.actuators.control_models import ControlModel


class ControlModelImpl(ControlModel):
    @override
    def set_target_value(self, value: float) -> None:
        pass

    @override
    def step(self) -> float:
        return 0.0


class TestControlModel:
    """Tests for the ControlModel abstract base class."""

    @pytest.mark.parametrize("method", ["set_target_value", "step"])
    def test_abstractmethods(self, method: str):
        """Test that ControlModel correctly defines expected abstract
        methods."""
        assert method in ControlModel.__abstractmethods__

    def test_init(self):
        """Test initialization with valid and invalid delta_time values."""

        # Test with valid delta_time
        model = ControlModelImpl(0.1)
        assert model.delta_time == 0.1

        # Test with invalid delta_time
        with pytest.raises(ValueError, match="delta_time must be larger than 0.0"):
            ControlModelImpl(0.0)

        with pytest.raises(ValueError, match="delta_time must be larger than 0.0"):
            ControlModelImpl(-1.0)

    def test_delta_time(self):
        """Test the delta_time property setter and getter."""
        model = ControlModelImpl(0.1)

        # Test getter
        assert model.delta_time == 0.1

        # Test setter with valid value
        model.delta_time = 0.2
        assert model.delta_time == 0.2

        # Test setter with invalid values
        with pytest.raises(ValueError, match="delta_time must be larger than 0.0"):
            model.delta_time = 0.0

        with pytest.raises(ValueError, match="delta_time must be larger than 0.0"):
            model.delta_time = -0.5

        # Ensure original value is preserved after failed attempt
        assert model.delta_time == 0.2
