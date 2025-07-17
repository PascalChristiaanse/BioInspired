"""This submodule provides all simulation environments and assembles the
simulation environment for the bioinspired package.
"""

from .empty_universe import EmptyUniverseSimulator
from .earth import EarthSimulator

# Ensure imports are available in the package namespace
__all__ = [
    "EmptyUniverseSimulator",
    "EarthSimulator",
]
