from typing import TypedDict, override

from pamiq_core.interaction.modular_env import Actuator
from pamiq_io.mouse import InputtinoMouseOutput, MouseButton

from .control_models import SimpleButton, SimpleMotor


class MouseAction(TypedDict, total=False):
    """Definition of possible mouse actions.

    Attributes:
        move_velocity: Tuple of (x, y) velocity in pixels per second.
        button_press: Dictionary mapping buttons to their press state (True for pressed, False for released).
    """

    move_velocity: tuple[float, float]
    button_press: dict[MouseButton, bool]


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
        for button in MouseButton:
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
            for button in MouseButton:
                self._mouse.release(button)


class SmoothMouseActuator(MouseActuator):
    """Mouse actuator that provides smooth motion and button transitions.

    This actuator extends the standard MouseActuator by adding gradual transitions
    to mouse movements and button actions, creating more natural-looking interactions.
    It uses control models (SimpleMotor and SimpleButton) to implement velocity ramping
    and delayed button responses.

    Examples:
        >>> actuator = SmoothMouseActuator(
        ...     delta_time=0.05,           # 50ms update interval
        ...     time_constant=0.2,         # 200ms velocity smoothing
        ...     press_delay=0.05,          # 50ms delay before press takes effect
        ...     release_delay=0.1          # 100ms delay before release takes effect
        ... )
        >>> # Motion will gradually accelerate to target velocity
        >>> actuator.operate({"move_velocity": (100.0, 50.0)})
        >>> # Buttons will have realistic press/release delays
        >>> actuator.operate({"button_press": {MouseButton.LEFT: True}})
    """

    @override
    def __init__(
        self,
        delta_time: float = 0.1,
        time_constant: float = 0.2,
        press_delay: float = 0.04,
        release_delay: float = 0.16,
    ) -> None:
        """Initialize the smooth mouse actuator.

        Args:
            delta_time: The time step for control model updates in seconds.
            time_constant: Time constant for velocity smoothing in seconds.
            press_delay: Delay before button press takes effect in seconds.
            release_delay: Delay before button release takes effect in seconds.
        """
        super().__init__()

        # Create velocity control models
        self._vx_motor = SimpleMotor(delta_time, time_constant)
        self._vy_motor = SimpleMotor(delta_time, time_constant)

        # Create button control models
        self._button_models = {
            button: SimpleButton(delta_time, False, press_delay, release_delay)
            for button in MouseButton
        }

    @override
    def operate(self, action: MouseAction) -> None:
        """Execute the specified mouse action with smooth transitions.

        Args:
            action: The mouse action to execute, containing velocity and/or button states.
        """
        if move_vel := action.get("move_velocity"):
            self._vx_motor.set_target_value(move_vel[0])
            self._vy_motor.set_target_value(move_vel[1])

        if buttons := action.get("button_press"):
            for btn, pressed in buttons.items():
                if btn in self._button_models:
                    self._button_models[btn].set_target_value(pressed)

        action = {
            "move_velocity": (self._vx_motor.step(), self._vy_motor.step()),
            "button_press": {k: bool(v.step()) for k, v in self._button_models.items()},
        }

        super().operate(action)
