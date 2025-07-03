import time
from enum import StrEnum
from typing import ClassVar, TypedDict, override

from pamiq_core import Actuator
from pamiq_io.osc import OscOutput

from .control_models import SimpleButton, SimpleMotor


class Axes(StrEnum):
    """Enumerates Axis control addresses.

    See  <https://docs.vrchat.com/docs/osc-as-input-controller#axes>
    """

    Vertical = "/input/Vertical"
    Horizontal = "/input/Horizontal"
    LookHorizontal = "/input/LookHorizontal"
    UseAxisRight = "/input/UseAxisRight"
    GrabAxisRight = "/input/GrabAxisRight"
    MoveHoldFB = "/input/MoveHoldFB"
    SpinHoldCwCcw = "/input/SpinHoldCwCcw"
    SpinHoldUD = "/input/SpinHoldUD"
    SpinHoldLR = "/input/SpinHoldLR"


class Buttons(StrEnum):
    """Enumerates Button control addresses.

    See
    <https://docs.vrchat.com/docs/osc-as-input-controller#buttons>
    """

    MoveForward = "/input/MoveForward"
    MoveBackward = "/input/MoveBackward"
    MoveLeft = "/input/MoveLeft"
    MoveRight = "/input/MoveRight"
    LookLeft = "/input/LookLeft"
    LookRight = "/input/LookRight"
    Jump = "/input/Jump"
    Run = "/input/Run"


RESET_COMMANDS = {str(addr): 0 for addr in Buttons} | {str(addr): 0.0 for addr in Axes}

type AxesAction = dict[Axes, float]
type ButtonsAction = dict[Buttons, bool]


class OscAction(TypedDict, total=False):
    axes: AxesAction
    buttons: ButtonsAction


class OscActuator(Actuator[OscAction]):
    """VRChat OSC-based actuator for controlling avatar movement and actions.

    This actuator allows interaction with a VRChat instance by sending OSC (Open Sound Control)
    messages to control avatar movement, looking direction, jumping, and other actions.

    It uses the standard VRChat OSC API endpoints for axes (continuous values) and buttons
    (binary values) to provide a comprehensive control interface.

    Examples:
        >>> actuator = OscActuator()
        >>> # Move forward at half speed
        >>> actuator.operate({"axes": {Axes.Vertical: 0.5}})
        >>> # Jump
        >>> actuator.operate({"buttons": {Buttons.Jump: True}})
        >>> # Move forward and run
        >>> actuator.operate({
        ...     "axes": {Axes.Vertical: 1.0},
        ...     "buttons": {Buttons.Run: True}
        ... })
    """

    CLOSE_MENU_COMMAND_DELAY: ClassVar[float] = 0.1 / 3
    DEFAULT_COMMAND_FOR_CLOSE_MENU = {str(Axes.Vertical): 1.0}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        command_for_close_menu: dict[str, float | int] = DEFAULT_COMMAND_FOR_CLOSE_MENU,
    ) -> None:
        """Initialize the OscActuator.

        Args:
            host: The IP address or hostname where VRChat is running.
            port: The port number VRChat is listening on for OSC messages.
            command_for_close_menu: OSC commands to send when closing VRChat menus.
                These commands help return to world interaction from menu states.
                Set to empty dict to disable this feature.
        """
        super().__init__()

        self._osc = OscOutput(host, port)
        self._command_for_close_menu = command_for_close_menu.copy()

        self._current_axes: AxesAction = {}
        self._current_buttons: ButtonsAction = {}

    @property
    def current_action(self) -> OscAction:
        """Get the current action state.

        Returns:
            A dictionary containing the current state of all axes and buttons.
        """

        return OscAction(axes=self._current_axes, buttons=self._current_buttons)

    @override
    def operate(self, action: OscAction) -> None:
        """Send OSC commands to VRChat based on the provided action.

        Only sends commands for values that have changed since the last operation,
        optimizing network traffic.

        Args:
            action: Dictionary containing axes and/or buttons to control.  Axes value should be in the range [-1.0, 1.0].

        Raises:
            ValueError: If any axis value is outside the valid range [-1.0, 1.0].
        """
        sending_commands: dict[str, float | int] = {}
        if axes := action.get("axes"):
            self.validate_axes(axes)
            for key, value in axes.items():
                if self._current_axes.get(key) != value:
                    self._current_axes[key] = value
                    sending_commands[key] = float(value)

        if buttons := action.get("buttons"):
            for key, value in buttons.items():
                if self._current_buttons.get(key) != value:
                    self._current_buttons[key] = value
                    sending_commands[key] = int(value)
        self._osc.send_messages(sending_commands)

    @staticmethod
    def validate_axes(axes: AxesAction) -> None:
        """Validate that all axis values are within the valid range.

        Args:
            axes: Dictionary mapping Axes enum values to float values.

        Raises:
            ValueError: If any axis value is outside the valid range [-1.0, 1.0].
        """
        for key, value in axes.items():
            if not (-1 <= value <= 1):
                raise ValueError(
                    f"Axes key must be in range [-1, 1], input: '{key}: {value}'"
                )

    def _close_menu(self) -> None:
        """Send a command for closing menu.

        This helps return to world interaction.
        """

        if self._command_for_close_menu == {}:
            return

        time.sleep(self.CLOSE_MENU_COMMAND_DELAY)
        self._osc.send_messages(self._command_for_close_menu)
        time.sleep(self.CLOSE_MENU_COMMAND_DELAY)
        self._osc.send_messages(RESET_COMMANDS)
        time.sleep(self.CLOSE_MENU_COMMAND_DELAY)

    @override
    def setup(self):
        """Initialize the actuator.

        Resets all controls to neutral positions and sends a jump
        command if jump_on_action_start is True.
        """
        super().setup()
        self._osc.send_messages(RESET_COMMANDS)
        self._close_menu()

    @override
    def teardown(self):
        """Clean up when the actuator is stopped.

        Resets all controls to neutral positions.
        """

        super().teardown()
        self._osc.send_messages(RESET_COMMANDS)

    @override
    def on_paused(self) -> None:
        """Handle system pause event.

        Resets all controls to neutral positions to prevent stuck
        inputs.
        """
        super().on_paused()
        self._osc.send_messages(RESET_COMMANDS)

    @override
    def on_resumed(self) -> None:
        """Handle system resume event.

        Sends a jump command if jump_on_action_start is True, then
        restores the previous action state.
        """
        super().on_resumed()
        self._close_menu()
        self.operate(self.current_action)


class SmoothOscActuator(OscActuator):
    """OSC actuator that provides smooth transitions for VRChat controls.

    This actuator extends the standard OscActuator by adding gradual transitions
    to axis movements and button actions, creating more natural-looking avatar
    movement and interactions. It uses control models (SimpleMotor and SimpleButton)
    to implement value ramping and delayed button responses.

    Examples:
        >>> actuator = SmoothOscActuator(
        ...     delta_time=0.05,           # 50ms update interval
        ...     time_constant=0.2,         # 200ms axis value smoothing
        ...     press_delay=0.05,          # 50ms delay before button press takes effect
        ...     release_delay=0.1          # 100ms delay before button release takes effect
        ... )
        >>> # Movement will gradually accelerate to target value
        >>> actuator.operate({"axes": {Axes.Vertical: 0.5}})
        >>> # Buttons will have realistic press/release delays
        >>> actuator.operate({"buttons": {Buttons.Jump: True}})
    """

    @override
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        command_for_close_menu: dict[
            str, float | int
        ] = OscActuator.DEFAULT_COMMAND_FOR_CLOSE_MENU,
        delta_time: float = 0.1,
        time_constant: float = 0.2,
        press_delay: float = 0.04,
        release_delay: float = 0.16,
    ) -> None:
        """Initialize the smooth OSC actuator.

        Args:
            host: The IP address or hostname where VRChat is running.
            port: The port number VRChat is listening on for OSC messages.
            command_for_close_menu: OSC commands to send when closing VRChat menus.
                These commands help return to world interaction from menu states.
                Set to empty dict to disable this feature.
            delta_time: The time step for control model updates in seconds.
            time_constant: Time constant for axis value smoothing in seconds.
            press_delay: Delay before button press takes effect in seconds.
            release_delay: Delay before button release takes effect in seconds.
        """
        super().__init__(host, port, command_for_close_menu)

        # Create control models for each axis
        self._axis_motors = {
            axis: SimpleMotor(delta_time, time_constant) for axis in Axes
        }

        # Create control models for each button
        self._button_models = {
            button: SimpleButton(delta_time, False, press_delay, release_delay)
            for button in Buttons
        }

    @override
    def operate(self, action: OscAction) -> None:
        """Execute the specified OSC action with smooth transitions.

        Args:
            action: The OSC action to execute, containing axes and/or button values.

        Raises:
            ValueError: If any axis value is outside the valid range [-1.0, 1.0].
        """
        # Process axes with smooth transitions
        if axes := action.get("axes"):
            self.validate_axes(axes)
            for axis, value in axes.items():
                self._axis_motors[axis].set_target_value(value)

        # Process buttons with delayed responses
        if buttons := action.get("buttons"):
            for button, pressed in buttons.items():
                self._button_models[button].set_target_value(pressed)

        # Construct smooth action
        smooth_action: OscAction = {
            "axes": {ax: motor.step() for ax, motor in self._axis_motors.items()},
            "buttons": {
                btn: bool(model.step()) for btn, model in self._button_models.items()
            },
        }

        super().operate(smooth_action)
