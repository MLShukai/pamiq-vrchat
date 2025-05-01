import pytest
from pytest_mock import MockerFixture

from pamiq_vrchat.actuators.mouse import MouseAction, MouseActuator, MouseButton


class TestMouseActuator:
    """Tests for the MouseActuator class."""

    @pytest.fixture
    def mock_mouse_output(self, mocker: MockerFixture):
        """Create a mock for InputtinoMouseOutput."""
        return mocker.patch("pamiq_vrchat.actuators.mouse.InputtinoMouseOutput")

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
