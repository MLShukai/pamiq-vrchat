import math
from typing import override

import pytest

from pamiq_vrchat.actuators.control_models import (
    ControlModel,
    SimpleButton,
    SimpleMotor,
)


class ControlModelImpl(ControlModel):
    @override
    def set_target_value(self, value: float) -> None:
        pass

    @property
    @override
    def current_value(self) -> float:
        return 0.5  # Return a fixed value for testing


class TestControlModel:
    """Tests for the ControlModel abstract base class."""

    @pytest.mark.parametrize("method_name", ["set_target_value", "current_value"])
    def test_abstract_methods(self, method_name: str):
        """Test that ControlModel correctly defines expected abstract
        methods."""
        assert method_name in ControlModel.__abstractmethods__

    def test_init(self):
        """Test initialization with valid and invalid delta_time values."""

        # Test with valid delta_time
        model = ControlModelImpl(0.1)
        assert model.delta_time == 0.1
        assert model.elapsed_time == 0.0  # Initial elapsed_time should be 0.0

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

    def test_elapsed_time_property(self):
        """Test the elapsed_time property."""
        model = ControlModelImpl(0.1)
        assert model.elapsed_time == 0.0

        # elapsed_time is a property and cannot be modified directly
        # use update() to modify it
        model.update()
        assert model.elapsed_time == 0.1

    def test_reset(self):
        """Test the reset method."""
        model = ControlModelImpl(0.1)

        # Increase elapsed_time
        model.update()
        model.update()
        assert model.elapsed_time == 0.2

        # reset() should set elapsed_time back to 0.0
        model.reset()
        assert model.elapsed_time == 0.0

    def test_update(self):
        """Test the update method."""
        model = ControlModelImpl(0.1)

        # Initial elapsed_time should be 0.0
        assert model.elapsed_time == 0.0

        # After update, elapsed_time should increase by delta_time
        elapsed_time = model.update()
        assert model.elapsed_time == 0.1
        assert elapsed_time == 0.1

        # Change delta_time and update again
        model.delta_time = 0.2
        elapsed_time = model.update()
        assert model.elapsed_time == pytest.approx(0.3)  # 0.1 + 0.2
        assert elapsed_time == model.elapsed_time

    def test_step(self):
        """Test the step method."""
        model = ControlModelImpl(0.1)

        # Step should update elapsed_time and return current_value
        result = model.step()
        assert model.elapsed_time == 0.1
        assert result == 0.5  # Fixed value from our ControlModelImpl

        # Second step
        result = model.step()
        assert model.elapsed_time == 0.2
        assert result == 0.5  # Same fixed value


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


class TestSimpleButton:
    """Tests for the SimpleButton class."""

    def test_init(self):
        """Test initialization with valid and invalid parameters."""
        # Test with valid parameters
        button = SimpleButton(
            delta_time=0.1, initial_state=False, push_delay=0.2, release_delay=0.3
        )
        assert button.delta_time == 0.1
        assert button.push_delay == 0.2
        assert button.release_delay == 0.3
        assert button.is_pressed is False
        assert button.current_value == 0.0

        # Test with initial_state=True
        button = SimpleButton(delta_time=0.1, initial_state=True)
        assert button.is_pressed is True
        assert button.current_value == 1.0

        # Test with invalid push_delay
        with pytest.raises(ValueError, match="push_delay must be non-negative"):
            SimpleButton(delta_time=0.1, push_delay=-0.1)

        # Test with invalid release_delay
        with pytest.raises(ValueError, match="release_delay must be non-negative"):
            SimpleButton(delta_time=0.1, release_delay=-0.1)

    def test_delay_property_setters(self):
        """Test the push_delay and release_delay property setters."""
        button = SimpleButton(delta_time=0.1)

        # Test push_delay setter
        assert button.push_delay == 0.0
        button.push_delay = 0.5
        assert button.push_delay == 0.5

        # Test invalid push_delay
        with pytest.raises(ValueError, match="push_delay must be non-negative"):
            button.push_delay = -0.1

        # Test release_delay setter
        assert button.release_delay == 0.0
        button.release_delay = 0.5
        assert button.release_delay == 0.5

        # Test invalid release_delay
        with pytest.raises(ValueError, match="release_delay must be non-negative"):
            button.release_delay = -0.1

    def test_immediate_state_change(self):
        """Test button state change with no delays."""
        button = SimpleButton(delta_time=0.1, push_delay=0.0, release_delay=0.0)

        # Button should start released
        assert button.is_pressed is False
        assert button.current_value == 0.0

        # Set target to pressed, should change immediately
        button.set_target_value(1.0)
        assert button.current_value == 1.0
        assert button.is_pressed is True

        # Set target to released, should change immediately
        button.set_target_value(0.0)
        assert button.current_value == 0.0
        assert button.is_pressed is False

    def test_delayed_press(self):
        """Test delayed button press."""
        push_delay = 0.3
        button = SimpleButton(delta_time=0.1, push_delay=push_delay, release_delay=0.0)

        # Set target to pressed
        button.set_target_value(1.0)

        # Button should still be released initially
        assert button.is_pressed is False
        assert button.current_value == 0.0

        # Take steps not reaching the push_delay
        button.step()  # elapsed_time = 0.1
        button.step()  # elapsed_time = 0.2
        assert button.is_pressed is False
        assert button.current_value == 0.0

        # Take step that reaches the push_delay
        button.step()  # elapsed_time = 0.3
        assert button.is_pressed is True
        assert button.current_value == 1.0

    def test_delayed_release(self):
        """Test delayed button release."""
        release_delay = 0.3
        button = SimpleButton(
            delta_time=0.1,
            initial_state=True,
            push_delay=0.0,
            release_delay=release_delay,
        )

        # Set target to released
        button.set_target_value(0.0)

        # Button should still be pressed initially
        assert button.is_pressed is True
        assert button.current_value == 1.0

        # Take steps not reaching the release_delay
        button.step()  # elapsed_time = 0.1
        button.step()  # elapsed_time = 0.2
        assert button.is_pressed is True
        assert button.current_value == 1.0

        # Take step that reaches the release_delay
        button.step()  # elapsed_time = 0.3
        assert button.is_pressed is False
        assert button.current_value == 0.0

    def test_value_conversion(self):
        """Test that non-zero values are treated as True for button press."""
        button = SimpleButton(delta_time=0.1)

        # Try different non-zero values
        for value in [0.5, 1, 42, -1.5]:
            button.set_target_value(value)
            assert button._target_pressed is True

        # Try zero value
        button.set_target_value(0.0)
        assert button._target_pressed is False
