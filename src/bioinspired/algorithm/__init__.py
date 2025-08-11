"""
Algorithm modules for evolutionary computation.
"""

from .sliceable_population import SliceablePopulation
from .tracked_algorithm import TrackedAlgorithm
from .tracked_archipelago import TrackedArchipelago

__all__ = [
    "SliceablePopulation",
    "TrackedAlgorithm",
    "TrackedArchipelago",
]
