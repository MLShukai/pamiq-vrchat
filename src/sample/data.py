from enum import StrEnum, auto


class BufferName(StrEnum):
    """Enumerates all buffer names in the experiments."""

    IMAGE = auto()


class DataKey(StrEnum):
    """Enumerates all data key names in the experiments."""

    IMAGE = auto()
