from importlib import metadata

from . import actuators, sensors
from .env import (
    ActionType,
    ObservationType,
    VRChatAction,
    VRChatEnvironment,
    VRChatObservation,
)

__version__ = metadata.version(__name__.replace("_", "-"))

__all__ = [
    "actuators",
    "sensors",
    "ActionType",
    "ObservationType",
    "VRChatAction",
    "VRChatObservation",
    "VRChatEnvironment",
]
