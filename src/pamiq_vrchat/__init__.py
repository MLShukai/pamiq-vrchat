from importlib import metadata

from . import actuators, sensors

__version__ = metadata.version(__name__.replace("_", "-"))

__all__ = ["actuators", "sensors"]
