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

    @abstractmethod
    def step(self) -> float:
        """Step the simulation forward by one time step.

        This method should update the internal state of the control
        system according to its dynamics and the current time step.

        Returns:
            Current output value of the system after the time step.
        """
        ...


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
        self._current_elapsed_time = 0.0

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
    def current_value(self) -> float:
        """Get the current output value of the motor.

        Returns:
            Current output value of the motor.
        """
        return self._calculate_current_value()

    def _calculate_current_value(self) -> float:
        """Calculate the current output value using the analytical solution.

        Calculates the current value using the analytical solution
        of the first-order differential equation:
            y(t) = y_0 + (u - y_0)(1 - e^(-t/τ))

        Returns:
            Current output value of the motor.
        """
        return self._start_value + (self._target_value - self._start_value) * (
            1 - math.exp(-self._current_elapsed_time / self._time_constant)
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
            self._current_elapsed_time = 0.0

        self._target_value = value

    @override
    def step(self) -> float:
        """Step the motor simulation forward by one time step.

        Updates the elapsed time and calculates the new output value
        using the analytical solution for the first-order system.

        Returns:
            Current output value of the motor after the time step.
        """
        self._current_elapsed_time += self.delta_time

        return self.current_value
