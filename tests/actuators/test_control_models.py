import math
from typing import override

import pytest

from pamiq_vrchat.actuators.control_models import ControlModel, SimpleMotor


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


class TestSimpleMotor:
    """Tests for the SimpleMotor class."""

    def test_init(self):
        """Test initialization with valid and invalid parameters."""
        # Test with valid parameters
        motor = SimpleMotor(delta_time=0.1, time_constant=0.5, initial_value=1.0)
        assert motor.delta_time == 0.1
        assert motor.time_constant == 0.5
        assert motor.current_value == 1.0

        # Test with invalid time_constant
        with pytest.raises(ValueError, match="time_constant must be larger than 0.0"):
            SimpleMotor(delta_time=0.1, time_constant=0.0)

        with pytest.raises(ValueError, match="time_constant must be larger than 0.0"):
            SimpleMotor(delta_time=0.1, time_constant=-1.0)

    def test_time_constant_property(self):
        """Test the time_constant property setter and getter."""
        motor = SimpleMotor(delta_time=0.1, time_constant=0.5)

        # Test getter
        assert motor.time_constant == 0.5

        # Test setter with valid value
        motor.time_constant = 1.0
        assert motor.time_constant == 1.0

        # Test setter with invalid values
        with pytest.raises(ValueError, match="time_constant must be larger than 0.0"):
            motor.time_constant = 0.0

        with pytest.raises(ValueError, match="time_constant must be larger than 0.0"):
            motor.time_constant = -0.5

        # Ensure original value is preserved after failed attempt
        assert motor.time_constant == 1.0

    def test_current_value_property(self):
        """Test the current_value property."""
        motor = SimpleMotor(delta_time=0.1, time_constant=0.5, initial_value=1.0)
        assert motor.current_value == 1.0

    def test_set_target_value(self):
        """Test setting a target value."""
        motor = SimpleMotor(delta_time=0.1, time_constant=0.5, initial_value=0.0)

        # Initial state
        assert motor.current_value == 0.0

        # Set target to 1.0
        motor.set_target_value(1.0)
        assert (
            motor.current_value == 0.0
        )  # Value shouldn't change until step() is called

        # Setting the same target value twice shouldn't reset the system
        initial_state = motor.step()
        motor.set_target_value(1.0)  # Same value
        assert motor.step() > initial_state  # Should continue from previous state

    def test_step(self):
        """Test stepping the simulation and approaching the target value."""
        delta_time = 0.1
        time_constant = 0.5
        motor = SimpleMotor(
            delta_time=delta_time, time_constant=time_constant, initial_value=0.0
        )

        # Set target to 1.0
        motor.set_target_value(1.0)

        # After one time constant (t = τ), value should be approximately 63.2% of target
        steps_for_one_time_constant = int(time_constant / delta_time)

        for _ in range(steps_for_one_time_constant):
            motor.step()

        # Check value is close to 63.2% of target (1 - e^-1 ≈ 0.632)
        expected_value = 1.0 * (1 - math.exp(-1))
        assert abs(motor.current_value - expected_value) < 1e-6

        # After three time constants (t = 3τ), value should be approximately 95% of target
        for _ in range(2 * steps_for_one_time_constant):  # 2 more time constants
            motor.step()

        # Check value is close to 95% of target (1 - e^-3 ≈ 0.95)
        expected_value = 1.0 * (1 - math.exp(-3))
        assert abs(motor.current_value - expected_value) < 1e-6

    def test_step_with_changing_target(self):
        """Test changing the target value during the simulation."""
        motor = SimpleMotor(delta_time=0.1, time_constant=0.5, initial_value=0.0)

        # Start approaching target=1.0
        motor.set_target_value(1.0)
        for _ in range(5):  # Take 5 steps
            motor.step()

        half_way_value = motor.current_value
        assert 0.0 < half_way_value < 1.0  # Should be between 0 and 1

        # Change target to -1.0
        motor.set_target_value(-1.0)

        # First step should start from the halfway point
        assert motor.step() < half_way_value  # Should decrease

        # Continue stepping until close to new target
        for _ in range(20):  # More than enough steps to get close to target
            motor.step()

        assert motor.current_value < -0.9  # Should be close to -1.0
