"""This module contains all spacecraft designs and configurations for the
bioinspired package. It contains:
 - A spacecraft design base class that can be used to create
   spacecraft designs.
    - A simplified spacecraft to test basic simulation functionality.
    - The Endurance spacecraft from the movie Interstellar.
    - The Lander 2 spacecraft from the movie Interstellar."""

from .simple_craft import SimpleCraft
from .lander_2 import Lander2
from .endurance import Endurance
from .spacecraft_base import SpacecraftBase
from .rotating_spacecraft_base import RotatingSpacecraftBase
from .JSON_spacecraft_base import JSONSpacecraftBase
from .ephemeris_spacecraft_base import EphemerisSpacecraftBase

__all__ = [
    "SimpleCraft",
    "Lander2",
    "Endurance",
    "SpacecraftBase",
    "RotatingSpacecraftBase",
    "JSONSpacecraftBase",
    "EphemerisSpacecraftBase",
]
