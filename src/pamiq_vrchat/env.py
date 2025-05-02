from collections.abc import Mapping
from enum import StrEnum, auto
from typing import Any, cast, override

from pamiq_core.interaction.modular_env import (
    Actuator,
    ActuatorsDict,
    ModularEnvironment,
    Sensor,
    SensorsDict,
)


class ObservationType(StrEnum):
    """Enumeration of observation types for VRChat environment.

    Defines the types of observations available from the VRChat
    environment.
    """

    IMAGE = auto()


class ActionType(StrEnum):
    """Enumeration of action types for VRChat environment.

    Defines the types of actions that can be performed in the VRChat
    environment.
    """

    OSC = auto()
    MOUSE = auto()


type VRChatObservation = Mapping[ObservationType, Any]
type VRChatAction = Mapping[ActionType, Any]


class VRChatEnvironment(ModularEnvironment[VRChatObservation, VRChatAction]):
    """Environment implementation for VRChat interaction.

    This class provides a modular environment for interacting with
    VRChat, supporting various sensors and actuators.

    The environment uses enum-based keys to organize sensors and
    actuators, allowing for type-safe access to observations and
    actions.
    """

    @override
    def __init__(
        self,
        sensors: Mapping[ObservationType, Sensor[Any]],
        actuators: Mapping[ActionType, Actuator[Any]],
    ) -> None:
        """Initialize the VRChat environment.

        Args:
            sensors: Mapping of observation types to sensor implementations.
            actuators: Mapping of action types to actuator implementations.
        """
        self.sensors = dict(sensors)
        self.actuators = dict(actuators)

        sensors_dict = SensorsDict({str(k): v for k, v in self.sensors.items()})
        actuators_dict = ActuatorsDict({str(k): v for k, v in actuators.items()})
        # Casting to avoid type error because Mapping[StrEnum, ...] is not compatible to Mapping[str, ...].
        super().__init__(
            cast(Sensor[Any], sensors_dict), cast(Actuator[Any], actuators_dict)
        )

    @override
    def observe(self) -> VRChatObservation:
        """Collect observations from all sensors.

        Returns:
            A mapping of observation types to sensor readings.
        """
        return {k: s.read() for k, s in self.sensors.items()}

    @override
    def affect(self, action: VRChatAction) -> None:
        """Apply actions to all actuators.

        Args:
            action: A mapping of action types to actuator commands.

        Raises:
            ValueError: If an action key specified in the environment's actuators
                is not present in the provided action mapping.
        """
        for k, a in self.actuators.items():
            if k not in action:
                raise ValueError(f"Action key {k} does not exists in action.")
            a.operate(action[k])
