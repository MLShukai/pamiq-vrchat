from importlib import metadata

from . import actuators, sensors
from .utils import (
    ActionType,
    ObservationType,
)

__version__ = metadata.version(__name__.replace("_", "-"))

__all__ = [
    "actuators",
    "sensors",
    "ActionType",
    "ObservationType",
]
