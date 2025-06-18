"""
BioInspired: Evolutionary algorithms for automated docking procedures.
"""

from .data import DatabaseManager, TrajectoryManager
from .data.models import Environment, Algorithm, Individual, Trajectory, Annotation

__version__ = "0.1.0"
__all__ = [
    "DatabaseManager", 
    "TrajectoryManager",
    "Environment", 
    "Algorithm", 
    "Individual", 
    "Trajectory", 
    "Annotation"
]
