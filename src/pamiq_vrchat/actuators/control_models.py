from abc import ABC, abstractmethod


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
