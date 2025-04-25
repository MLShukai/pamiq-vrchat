from typing import TypedDict, override

from pamiq_core.interaction.modular_env import Actuator
from pamiq_io.mouse import Button, ButtonLiteral, InputtinoMouseOutput


class MouseAction(TypedDict, total=False):
    """Definition of possible mouse actions.

    Attributes:
        move_velocity: Tuple of (x, y) velocity in pixels per second.
        button_press: Dictionary mapping buttons to their press state (True for pressed, False for released).
    """

    move_velocity: tuple[float, float]
    button_press: dict[Button, bool]


# Define constants for button names
MOUSE_BUTTONS: list[ButtonLiteral] = ["left", "right", "middle", "side", "extra"]


class MouseActuator(Actuator[MouseAction]):
    """Actuator for controlling mouse movements and button presses.

    This actuator translates high-level mouse actions into physical
    mouse movements and button presses using InputtinoMouseOutput.
    """

    def __init__(self) -> None:
        """Initialize the mouse actuator.

        Creates an instance of InputtinoMouseOutput to handle the
        physical mouse control.
        """
        super().__init__()

        self._mouse = InputtinoMouseOutput()
        self._current_action: MouseAction | None = None

    @override
    def operate(self, action: MouseAction) -> None:
        """Execute the specified mouse action.

        Args:
            action: The mouse action to execute, containing velocity and/or button states.
        """
        move_vel = action.get("move_velocity")
        if move_vel is not None:
            self._mouse.move(move_vel[0], move_vel[1])

        buttons = action.get("button_press")
        if buttons is not None:
            for btn, pressed in buttons.items():
                if pressed:
                    self._mouse.press(btn)
                else:
                    self._mouse.release(btn)

        self._current_action = action

    @override
    def on_paused(self) -> None:
        """Handle system pause event.

        Stops all mouse movement and releases all buttons.
        """
        super().on_paused()
        self._mouse.move(0, 0)  # Stop mouse movement.

        # Release all buttons
        for button in MOUSE_BUTTONS:
            self._mouse.release(button)

    @override
    def on_resumed(self) -> None:
        """Handle system resume event.

        Restores the last mouse action that was being executed before
        pausing.
        """
        super().on_resumed()
        if self._current_action is not None:
            self.operate(self._current_action)

    def __del__(self) -> None:
        """Clean up resources when the object is destroyed.

        Ensures that mouse movement is stopped and all buttons are
        released.
        """
        if hasattr(self, "_mouse"):
            self._mouse.move(0, 0)
            for button in MOUSE_BUTTONS:
                self._mouse.release(button)
