from enum import StrEnum


class ObservationType(StrEnum):
    """Enumeration of observation types for VRChat environment.

    Defines the types of observations available from the VRChat
    environment.
    """

    IMAGE = "image"
    AUDIO = "audio"


class ActionType(StrEnum):
    """Enumeration of action types for VRChat environment.

    Defines the types of actions that can be performed in the VRChat
    environment.
    """

    OSC = "osc"
    MOUSE = "mouse"
