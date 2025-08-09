"""
Algorithm modules for evolutionary computation.
"""

from .initializer_BFE import InitializerBFE
from .sliceable_population import SliceablePopulation

__all__ = [
    "InitializerBFE",
    "SliceablePopulation", 
]