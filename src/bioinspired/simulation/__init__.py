"""This submodule provides all simulation environments and assembles the
simulation environment for the bioinspired package.
"""

from .empty_universe import EmptyUniverseSimulator

# Ensure imports are available in the package namespace
__all__ = [
    "EmptyUniverseSimulator",
]
