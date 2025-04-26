"""Tests for the OscActuator class."""

import pytest
from pytest_mock import MockerFixture

from pamiq_vrchat.actuators.osc import (
    RESET_COMMANDS,
    Axes,
    Buttons,
    OscAction,
    OscActuator,
)


class TestOscActuator:
    """Tests for the OscActuator class."""

    @pytest.fixture
    def mock_osc_output_cls(self, mocker: MockerFixture):
        """Create a mock for the OscOutput class."""
        return mocker.patch("pamiq_vrchat.actuators.osc.OscOutput", autospec=True)

    @pytest.fixture
    def mock_osc_output(self, mock_osc_output_cls):
        """Create a mock for the OscOutput instance."""
        return mock_osc_output_cls.return_value

    @pytest.fixture
    def mock_time_sleep(self, mocker: MockerFixture):
        """Mock time.sleep to avoid actual sleeping."""
        return mocker.patch("time.sleep")

    @pytest.fixture
    def actuator(self, mock_osc_output, mock_time_sleep):
        """Create an OscActuator instance for testing."""
        return OscActuator()

    def test_init_default_parameters(self, mock_osc_output_cls):
        """Test initialization with default parameters."""
        actuator = OscActuator()

        # Check that OscOutput was created with initial parameters
        mock_osc_output_cls.assert_called_once_with("127.0.0.1", 9000)

        # Verify default attribute values
        assert actuator.jump_on_action_start is True
        assert actuator.current_action == {"axes": {}, "buttons": {}}

    def test_init_custom_parameters(self, mock_osc_output_cls):
        """Test initialization with custom parameters."""
        custom_host = "192.168.1.100"
        custom_port = 8000
        jump_on_start = False

        actuator = OscActuator(
            host=custom_host,
            port=custom_port,
            jump_on_action_start=jump_on_start,
        )

        # Check that OscOutput was created with custom parameters
        mock_osc_output_cls.assert_called_once_with(custom_host, custom_port)

        # Verify custom attribute values
        assert actuator.jump_on_action_start is jump_on_start

    def test_current_action_property(self, actuator):
        """Test the current_action property."""
        # Set state using public operate method
        actuator.operate(
            OscAction(axes={Axes.Vertical: 0.5}, buttons={Buttons.Jump: True})
        )

        # Get the current action
        action = actuator.current_action

        # Verify the action contains the expected values
        assert isinstance(action, dict)
        assert "axes" in action
        assert "buttons" in action
        assert action["axes"] == {Axes.Vertical: 0.5}
        assert action["buttons"] == {Buttons.Jump: True}

    def test_operate_with_axes(self, actuator, mock_osc_output):
        """Test operate method with axes commands."""
        # Create an action with axes values
        action = OscAction(
            axes={
                Axes.Vertical: 0.5,
                Axes.Horizontal: -0.5,
            }
        )

        # Call operate
        actuator.operate(action)

        # Verify that send_messages was called with the right parameters
        mock_osc_output.send_messages.assert_called_once_with(
            {
                Axes.Vertical: 0.5,
                Axes.Horizontal: -0.5,
            }
        )

        # Verify state using public current_action property
        assert actuator.current_action["axes"] == {
            Axes.Vertical: 0.5,
            Axes.Horizontal: -0.5,
        }

    def test_operate_with_buttons(self, actuator, mock_osc_output):
        """Test operate method with button commands."""
        # Create an action with button values
        action = OscAction(
            buttons={
                Buttons.Jump: True,
                Buttons.Run: True,
            }
        )

        # Call operate
        actuator.operate(action)

        # Verify that send_messages was called with the right parameters
        mock_osc_output.send_messages.assert_called_once_with(
            {
                Buttons.Jump: 1,
                Buttons.Run: 1,
            }
        )

        # Verify state using public current_action property
        assert actuator.current_action["buttons"] == {
            Buttons.Jump: True,
            Buttons.Run: True,
        }

    def test_operate_with_both_axes_and_buttons(self, actuator, mock_osc_output):
        """Test operate method with both axes and button commands."""
        # Create an action with both axes and button values
        action = OscAction(
            axes={Axes.Vertical: 0.5},
            buttons={Buttons.Run: True},
        )

        # Call operate
        actuator.operate(action)

        # Verify that send_messages was called with the right parameters
        mock_osc_output.send_messages.assert_called_once_with(
            {
                Axes.Vertical: 0.5,
                Buttons.Run: 1,
            }
        )

        # Verify state using public current_action property
        assert actuator.current_action["axes"] == {Axes.Vertical: 0.5}
        assert actuator.current_action["buttons"] == {Buttons.Run: True}

    def test_operate_unchanged_values_not_sent(self, actuator, mock_osc_output):
        """Test that operate doesn't send unchanged values."""
        # Set initial state using public operate method
        actuator.operate(
            OscAction(axes={Axes.Vertical: 0.5}, buttons={Buttons.Run: True})
        )

        # Create an action with the same values
        action = OscAction(
            axes={Axes.Vertical: 0.5},
            buttons={Buttons.Run: True},
        )

        # Reset the mock to clear previous calls
        mock_osc_output.send_messages.reset_mock()

        # Call operate
        actuator.operate(action)

        # Verify that send_messages was called with empty dict (no changes)
        mock_osc_output.send_messages.assert_called_once_with({})

    def test_validate_axes_valid_values(self, actuator):
        """Test validate_axes with valid axis values."""
        # Test with various valid values
        valid_axes = {
            Axes.Vertical: 0.0,
            Axes.Horizontal: 1.0,
            Axes.LookHorizontal: -1.0,
            Axes.UseAxisRight: 0.5,
        }

        # This should not raise an exception
        actuator.validate_axes(valid_axes)

    def test_validate_axes_invalid_values(self, actuator):
        """Test validate_axes with invalid axis values."""
        # Test with various invalid values
        invalid_axes = {
            Axes.Vertical: 1.1,  # Greater than 1.0
        }

        # This should raise a ValueError
        with pytest.raises(ValueError):
            actuator.validate_axes(invalid_axes)

        # Test another invalid value
        invalid_axes = {
            Axes.Horizontal: -1.1,  # Less than -1.0
        }

        # This should also raise a ValueError
        with pytest.raises(ValueError):
            actuator.validate_axes(invalid_axes)

    def test_setup_with_jump(self, actuator, mock_osc_output, mock_time_sleep):
        """Test setup with jump_on_action_start=True."""
        # Call setup
        actuator.setup()

        # Verify reset commands sent first
        mock_osc_output.send_messages.assert_called_once_with(RESET_COMMANDS)

        # Verify the jump sequence was sent (check the order and values of the calls)
        mock_osc_output.send.assert_any_call(Buttons.Jump, 1)
        mock_osc_output.send.assert_any_call(Buttons.Jump, 0)

        # Verify time.sleep was called appropriate times
        assert mock_time_sleep.call_count == 3

    def test_setup_without_jump(self, mock_osc_output, mock_time_sleep):
        """Test setup with jump_on_action_start=False."""
        # Create actuator with jump_on_action_start=False
        actuator = OscActuator(jump_on_action_start=False)

        # Call setup
        actuator.setup()

        # Verify reset commands sent
        mock_osc_output.send_messages.assert_called_once_with(RESET_COMMANDS)

        # Verify jump commands were not sent and no sleep was performed
        mock_osc_output.send.assert_not_called()
        mock_time_sleep.assert_not_called()

    def test_teardown(self, actuator, mock_osc_output):
        """Test the teardown method."""
        # Call teardown
        actuator.teardown()

        # Verify reset commands sent
        mock_osc_output.send_messages.assert_called_once_with(RESET_COMMANDS)

    def test_on_paused(self, actuator, mock_osc_output):
        """Test the on_paused method."""
        # Call on_paused
        actuator.on_paused()

        # Verify reset commands sent
        mock_osc_output.send_messages.assert_called_once_with(RESET_COMMANDS)

    def test_on_resumed_with_jump(
        self, actuator, mock_osc_output, mock_time_sleep, mocker: MockerFixture
    ):
        """Test on_resumed with jump_on_action_start=True."""
        # Set current action state
        test_action = OscAction(axes={Axes.Vertical: 0.5}, buttons={Buttons.Run: True})
        mock_operate = mocker.spy(actuator, "operate")
        actuator.operate(test_action)

        mock_operate.reset_mock()

        # Call on_resumed
        actuator.on_resumed()

        # Verify the jump sequence was sent
        mock_osc_output.send.assert_any_call(Buttons.Jump, 1)
        mock_osc_output.send.assert_any_call(Buttons.Jump, 0)

        # Verify time.sleep was called appropriate times
        assert mock_time_sleep.call_count == 3

        # Verify previous state was restored (should be sent after jump sequence)
        mock_operate.assert_called_with(test_action)

    def test_on_resumed_without_jump(
        self, mock_osc_output, mock_time_sleep, mocker: MockerFixture
    ):
        """Test on_resumed with jump_on_action_start=False."""
        # Create actuator with jump_on_action_start=False
        actuator = OscActuator(jump_on_action_start=False)
        mock_operate = mocker.spy(actuator, "operate")

        # Set current action state
        test_action = OscAction(axes={Axes.Vertical: 0.5}, buttons={Buttons.Run: True})
        actuator.operate(test_action)
        mock_operate.reset_mock()

        # Call on_resumed
        actuator.on_resumed()

        # Verify no jump commands were sent and no sleep was performed
        mock_osc_output.send.assert_not_called()
        mock_time_sleep.assert_not_called()

        mock_operate.assert_called_with(test_action)
