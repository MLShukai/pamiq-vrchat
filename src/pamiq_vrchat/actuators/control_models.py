import math
from abc import ABC, abstractmethod
from typing import override


class ControlModel(ABC):
    """Abstract base class for control system models.

    This class defines the interface for various control system models such as
    first-order delay systems, PID controllers, or other dynamic response models.
    Concrete implementations should inherit from this class and implement the
    abstract methods.

    The class provides a common time-step based simulation approach where each
    call to `step()` advances the simulation by the specified delta time.

    Attributes:
        delta_time: Time step for state updates in seconds.
        elapsed_time: Total simulation time elapsed since the last reset in seconds.
    """

    def __init__(self, delta_time: float) -> None:
        """Initialize the control model.

        Args:
            delta_time: Time step for state updates in seconds.
                Must be positive.

        Raises:
            ValueError: If delta_time is not positive.
        """
        self.delta_time = delta_time
        self._elapsed_time = 0.0

    @property
    def delta_time(self) -> float:
        """Get the current time step value.

        Returns:
            The time step in seconds.
        """
        return self._delta_time

    @delta_time.setter
    def delta_time(self, v: float) -> None:
        """Set the time step value.

        Args:
            v: New time step value in seconds.

        Raises:
            ValueError: If the value is not positive.
        """
        if v <= 0.0:
            raise ValueError("delta_time must be larger than 0.0")
        self._delta_time = v

    @property
    def elapsed_time(self) -> float:
        """Get the elapsed simulation time since the last reset.

        Returns:
            The elapsed time in seconds.
        """
        return self._elapsed_time

    def reset(self) -> None:
        """Reset the elapsed time counter to zero.

        This method is typically called when the target value changes to
        restart the timing for the new response curve.
        """
        self._elapsed_time = 0.0

    @abstractmethod
    def set_target_value(self, value: float) -> None:
        """Set a new target value for the control system.

        This method should update the system's target value and
        make any necessary state changes to properly model the
        system response to the new target.

        Args:
            value: New target value for the system.
        """
        ...

    @property
    @abstractmethod
    def current_value(self) -> float:
        """Get the current output value of the control system.

        This method should calculate and return the current output value
        of the control system based on its internal state and elapsed time.

        Returns:
            Current output value of the system.
        """
        ...

    def update(self) -> float:
        """Update the internal state of the control system.

        This method advances the simulation by one time step by updating
        the elapsed time counter. Subclasses may override this method to
        add additional state updates.

        Returns:
            The updated elapsed time.
        """
        self._elapsed_time += self.delta_time
        return self.elapsed_time

    def step(self) -> float:
        """Step the simulation forward by one time step.

        This method updates the internal state of the control
        system by calling update() and then returns the current value.
        Subclasses typically don't need to override this method unless
        they require special stepping behavior.

        Returns:
            Current output value of the system after the time step.
        """
        self.update()
        return self.current_value


class SimpleMotor(ControlModel):
    """A simple motor model using first-order delay system dynamics.

    This class simulates a simple motor with first-order delay dynamics,
    gradually approaching a target value according to the time constant.
    The motor's response follows the standard first-order system equation:
        τ * (dy/dt) + y = u
    where:
        τ: time constant
        y: output value (motor position/speed)
        u: input value (target position/speed)

    Attributes:
        delta_time: Time step for state updates in seconds.
        time_constant: Time constant (τ) of the motor in seconds.
    """

    def __init__(
        self, delta_time: float, time_constant: float, initial_value: float = 0.0
    ) -> None:
        """Initialize the simple motor model.

        Args:
            delta_time: Time step for state updates in seconds.
                Must be positive.
            time_constant: Time constant (τ) of the motor in seconds.
                Must be positive. Larger values result in slower response.
            initial_value: Initial output value of the motor.
                Defaults to 0.0.

        Raises:
            ValueError: If delta_time or time_constant is not positive.
        """
        super().__init__(delta_time)

        if time_constant <= 0.0:
            raise ValueError("time_constant must be larger than 0.0")

        self._time_constant = time_constant
        self._target_value = initial_value
        self._start_value = initial_value

    @property
    def time_constant(self) -> float:
        """Get the current time constant value.

        Returns:
            The time constant in seconds.
        """
        return self._time_constant

    @time_constant.setter
    def time_constant(self, value: float) -> None:
        """Set the time constant value.

        Args:
            value: New time constant value in seconds.

        Raises:
            ValueError: If the value is not positive.
        """
        if value <= 0.0:
            raise ValueError("time_constant must be larger than 0.0")
        self._time_constant = value

    @property
    @override
    def current_value(self) -> float:
        """Calculate the current output value using the analytical solution.

        Calculates the current value using the analytical solution
        of the first-order differential equation:
            y(t) = y_0 + (u - y_0)(1 - e^(-t/τ))

        Returns:
            Current output value of the motor.
        """
        return self._start_value + (self._target_value - self._start_value) * (
            1 - math.exp(-self.elapsed_time / self._time_constant)
        )

    @override
    def set_target_value(self, value: float) -> None:
        """Set a new target value for the motor.

        When the target value changes, the motor's response will start
        from the current value and exponentially approach the new target.
        The elapsed time counter is reset to maintain the correct
        exponential response from the current state.

        Args:
            value: New target value for the motor to approach.
        """
        if self._target_value != value:
            self._start_value = self.current_value
            self.reset()

        self._target_value = value


class SimpleButton(ControlModel):
    """A simple button model that simulates pressing and releasing actions with
    configurable delays.

    This class simulates a button that can be in either pressed (1.0) or released (0.0) state,
    with configurable delays for transitioning between states.

    Attributes:
        delta_time: Time step for state updates in seconds.
        push_delay: Delay before the button is considered pressed after being instructed to press.
        release_delay: Delay before the button is considered released after being instructed to release.
    """

    def __init__(
        self,
        delta_time: float,
        initial_state: bool = False,
        push_delay: float = 0.0,
        release_delay: float = 0.0,
    ) -> None:
        """Initialize the simple button model.

        Args:
            delta_time: Time step for state updates in seconds.
                Must be positive.
            initial_state: Initial state of the button (True for pressed, False for released).
                Defaults to False (released).
            push_delay: Delay in seconds before the button actually gets pressed after
                receiving the command to press. Must be non-negative.
            release_delay: Delay in seconds before the button actually gets released after
                receiving the command to release. Must be non-negative.

        Raises:
            ValueError: If delta_time is not positive, or if push_delay or release_delay is negative.
        """
        super().__init__(delta_time)

        self._is_pressed = bool(initial_state)
        self._target_pressed = bool(initial_state)
        self.push_delay = push_delay
        self.release_delay = release_delay

    @property
    def push_delay(self) -> float:
        """Get the current push delay value.

        Returns:
            The push delay in seconds.
        """
        return self._push_delay

    @push_delay.setter
    def push_delay(self, value: float) -> None:
        """Set the push delay value.

        Args:
            value: New push delay value in seconds.

        Raises:
            ValueError: If the value is negative.
        """
        if value < 0.0:
            raise ValueError("push_delay must be non-negative")
        self._push_delay = value

    @property
    def release_delay(self) -> float:
        """Get the current release delay value.

        Returns:
            The release delay in seconds.
        """
        return self._release_delay

    @release_delay.setter
    def release_delay(self, value: float) -> None:
        """Set the release delay value.

        Args:
            value: New release delay value in seconds.

        Raises:
            ValueError: If the value is negative.
        """
        if value < 0.0:
            raise ValueError("release_delay must be non-negative")
        self._release_delay = value

    @property
    def is_pressed(self) -> bool:
        """Get the current pressed state of the button.

        Returns:
            True if the button is currently pressed, False otherwise.
        """
        return self._is_pressed

    @override
    def set_target_value(self, value: float) -> None:
        """Set a new target state for the button.

        The button will transition to the new state after the appropriate delay
        (push_delay or release_delay).

        Args:
            value: New target value for the button. Non-zero values are interpreted
                as "pressed" (True), zero as "released" (False).
        """
        new_target = bool(value)
        if self._target_pressed != new_target:
            self._target_pressed = new_target
            self.reset()

    @property
    @override
    def current_value(self) -> float:
        """Calculate the current value of the button.

        Checks if the button should change state based on the elapsed time
        and configured delays. Returns 1.0 for pressed state, 0.0 for released state.

        Returns:
            1.0 if the button is currently pressed, 0.0 if released.
        """
        # If target and current state are already the same, no need to check delays
        if self._target_pressed == self._is_pressed:
            return float(self.is_pressed)

        # If target is pressed but button is not pressed yet
        if self._target_pressed and not self._is_pressed:
            if self.elapsed_time >= self._push_delay:
                self._is_pressed = True

        # If target is released but button is still pressed
        elif not self._target_pressed and self._is_pressed:
            if self.elapsed_time >= self._release_delay:
                self._is_pressed = False

        return float(self.is_pressed)
