from importlib.metadata import version

from . import dataloaders, experimental

__all__ = ["dataloaders", "experimental"]

__version__ = version("spatialstream")
