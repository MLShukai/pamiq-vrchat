from importlib import metadata

__version__ = metadata.version(__name__.replace("_", "-"))
