import time
from enum import StrEnum
from typing import ClassVar, TypedDict, override

from pamiq_core.interaction.modular_env import Actuator
from pamiq_io.osc import OscOutput


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

    JUMP_DELAY: ClassVar[float] = 0.1 / 3

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        jump_on_action_start: bool = True,
    ) -> None:
        """Initialize the OscActuator.

        Args:
            host: The IP address or hostname where VRChat is running.
            port: The port number VRChat is listening on for OSC messages.
            jump_on_action_start: Whether to automatically send a jump command when
                the actuator starts or resumes.  This helps close VRChat menus and
                return to world interaction.
        """
        super().__init__()

        self._osc = OscOutput(host, port)
        self.jump_on_action_start = jump_on_action_start  # これはVRChatのメニューを自動的に閉じるためにあるオプション。OSCコマンドで何か操作をすると勝手にメニューが閉じ、ワールド操作に移れる。

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
                    sending_commands[key] = value

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

    def _send_jump(self) -> None:
        """Send a jump command sequence with appropriate timing.

        This helps close VRChat menus and return to world interaction.
        """
        time.sleep(self.JUMP_DELAY)
        self._osc.send(Buttons.Jump, 1)
        time.sleep(self.JUMP_DELAY)
        self._osc.send(Buttons.Jump, 0)
        time.sleep(self.JUMP_DELAY)

    @override
    def setup(self):
        """Initialize the actuator.

        Resets all controls to neutral positions and sends a jump
        command if jump_on_action_start is True.
        """
        super().setup()
        self._osc.send_messages(RESET_COMMANDS)

        if self.jump_on_action_start:
            self._send_jump()

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
        if self.jump_on_action_start:
            self._send_jump()

        self.operate(self.current_action)
