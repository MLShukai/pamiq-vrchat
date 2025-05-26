import sys

import pytest
from pytest_mock import MockerFixture

from pamiq_vrchat.actuators.mouse import (
    MouseAction,
    MouseActuator,
    MouseButton,
    SmoothMouseActuator,
)

if sys.platform != "win32" and sys.platform != "linux":
    pytest.skip(f"{sys.platform} is not supported.", allow_module_level=True)


class TestMouseActuator:
    """Tests for the MouseActuator class."""

    @pytest.fixture
    def mock_mouse_output(self, mocker: MockerFixture):
        """Create a mock for InputtinoMouseOutput."""
        if sys.platform == "linux":
            return mocker.patch("pamiq_io.mouse.InputtinoMouseOutput")
        elif sys.platform == "win32":
            return mocker.patch("pamiq_io.mouse.WindowsMouseOutput")

    def test_init(self, mock_mouse_output):
        """Test the initialization of MouseActuator."""
        MouseActuator()

        # Verify InputtinoMouseOutput was created
        mock_mouse_output.assert_called_once()

    def test_operate_move(self, mock_mouse_output):
        """Test operating the actuator with move velocity."""
        mock_instance = mock_mouse_output.return_value

        actuator = MouseActuator()
        action: MouseAction = {"move_velocity": (100.0, 50.0)}

        actuator.operate(action)

        # Verify move was called with correct parameters
        mock_instance.move.assert_called_once_with(100.0, 50.0)

    def test_operate_buttons(self, mock_mouse_output):
        """Test operating the actuator with button presses."""
        mock_instance = mock_mouse_output.return_value

        actuator = MouseActuator()
        action: MouseAction = {
            "button_press": {MouseButton.LEFT: True, MouseButton.RIGHT: False}
        }

        actuator.operate(action)

        # Verify press and release were called appropriately
        mock_instance.press.assert_called_once_with(MouseButton.LEFT)
        mock_instance.release.assert_called_once_with(MouseButton.RIGHT)

    def test_on_paused(self, mock_mouse_output):
        """Test the behavior when system is paused."""
        mock_instance = mock_mouse_output.return_value

        actuator = MouseActuator()
        actuator.on_paused()

        # Verify motion was stopped
        mock_instance.move.assert_called_once_with(0, 0)

        # Verify all buttons were released
        assert mock_instance.release.call_count == len(MouseButton)
        for button in MouseButton:
            mock_instance.release.assert_any_call(button)

    def test_on_resumed(self, mock_mouse_output, mocker: MockerFixture):
        """Test the behavior when system is resumed."""
        actuator = MouseActuator()

        # Set a current action
        action: MouseAction = {"move_velocity": (100.0, 50.0)}
        actuator.operate(action)

        # Create a spy for operate method
        operate_spy = mocker.spy(actuator, "operate")

        actuator.on_resumed()

        # Verify operate was called with the current action
        operate_spy.assert_called_once_with(action)

    def test_on_resumed_with_no_action(self, mock_mouse_output, mocker: MockerFixture):
        """Test the behavior when system is resumed."""
        actuator = MouseActuator()

        # Create a spy for operate method
        operate_spy = mocker.spy(actuator, "operate")

        actuator.on_resumed()

        # Verify operate was called with the current action
        operate_spy.assert_not_called()


class TestSmoothMouseActuator:
    """Tests for the SmoothMouseActuator class."""

    def test_subclass(self):
        assert issubclass(SmoothMouseActuator, MouseActuator)

    @pytest.fixture
    def mock_mouse_output(self, mocker: MockerFixture):
        """Create a mock for InputtinoMouseOutput."""
        if sys.platform == "linux":
            return mocker.patch("pamiq_io.mouse.InputtinoMouseOutput")
        elif sys.platform == "win32":
            return mocker.patch("pamiq_io.mouse.WindowsMouseOutput")

    def test_init(self, mock_mouse_output):
        """Test the initialization of SmoothMouseActuator."""
        # Verify the constructor runs without errors
        SmoothMouseActuator()
        # Verify InputtinoMouseOutput was created
        mock_mouse_output.assert_called_once()

    def test_smooth_movement(self, mock_mouse_output):
        """Test that mouse movement is smoothly transitioned."""
        mock_instance = mock_mouse_output.return_value

        # Reset the mock to clear initialization calls
        mock_instance.move.reset_mock()

        actuator = SmoothMouseActuator(delta_time=0.01, time_constant=0.1)
        target_velocity = (100.0, 50.0)

        # First call should start with small velocity
        actuator.operate({"move_velocity": target_velocity})

        # Verify the first velocity is smaller than target (due to smoothing)
        first_call = mock_instance.move.call_args[0]
        assert 0 < first_call[0] < target_velocity[0]
        assert 0 < first_call[1] < target_velocity[1]

        # Reset mock to track only new calls
        mock_instance.move.reset_mock()

        # Multiple calls should gradually approach the target
        prev_vx, prev_vy = first_call

        # Call operate several times without changing target
        for _ in range(100):
            actuator.operate({})

            # Get latest velocity
            latest_call = mock_instance.move.call_args[0]
            current_vx, current_vy = latest_call

            # Verify movement is approaching target
            if prev_vx < target_velocity[0]:
                assert current_vx > prev_vx
            if prev_vy < target_velocity[1]:
                assert current_vy > prev_vy

            # Update previous values
            prev_vx, prev_vy = current_vx, current_vy

            # Reset mock to isolate next call
            mock_instance.move.reset_mock()

        # After several iterations, we should be close to target
        assert abs(prev_vx - target_velocity[0]) < 10.0
        assert abs(prev_vy - target_velocity[1]) < 5.0

    def test_smooth_button_transitions(self, mock_mouse_output, mocker: MockerFixture):
        """Test that button presses/releases have smooth transitions."""
        mock_instance = mock_mouse_output.return_value

        # Reset the mock to clear initialization calls
        mock_instance.press.reset_mock()
        mock_instance.release.reset_mock()

        # Create actuator with minimal delays for testing
        actuator = SmoothMouseActuator(
            delta_time=0.01, press_delay=0.05, release_delay=0.05
        )

        # Initially, button should not be pressed
        actuator.operate({"button_press": {MouseButton.LEFT: True}})

        # First step should not result in button press due to delay
        mock_instance.press.assert_not_called()

        # After several steps, the button should be pressed
        for _ in range(10):
            actuator.operate({})

        # Verify the button was eventually pressed
        mock_instance.press.assert_called_with(MouseButton.LEFT)

        # Reset mocks for testing release
        mock_instance.press.reset_mock()
        mock_instance.release.reset_mock()

        # Now test button release - should also have delay
        actuator.operate({"button_press": {MouseButton.LEFT: False}})

        # First step should not result in immediate release
        assert mocker.call(MouseButton.LEFT) not in mock_instance.release.call_args_list

        # After several steps, the button should be released
        for i in range(20):
            actuator.operate({})

        # Verify the button was eventually released
        assert mocker.call(MouseButton.LEFT) in mock_instance.release.call_args_list

    def test_combined_movement_and_buttons(self, mock_mouse_output):
        """Test simultaneous movement and button operations."""
        mock_instance = mock_mouse_output.return_value

        # Reset mocks
        mock_instance.move.reset_mock()
        mock_instance.press.reset_mock()

        actuator = SmoothMouseActuator(
            delta_time=0.01, time_constant=0.1, press_delay=0.05
        )

        # Set both velocity and button state
        actuator.operate(
            {"move_velocity": (50.0, 25.0), "button_press": {MouseButton.LEFT: True}}
        )

        # First step should have moved but not pressed
        assert mock_instance.move.called
        assert not mock_instance.press.called

        # Movement should start small due to smoothing
        first_move = mock_instance.move.call_args[0]
        assert 0 < first_move[0] < 50.0
        assert 0 < first_move[1] < 25.0

        # Reset mocks for next stage
        mock_instance.move.reset_mock()

        # After multiple steps, velocity should increase and button should press
        for _ in range(20):
            actuator.operate({})

            # Check when button gets pressed
            if mock_instance.press.called:
                mock_instance.press.assert_called_once_with(MouseButton.LEFT)
                break

        # Verify button was eventually pressed
        mock_instance.press.assert_called_once_with(MouseButton.LEFT)
