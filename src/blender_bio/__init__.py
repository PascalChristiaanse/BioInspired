"""
Blender Bio: 3D Visualization and Rendering for BioInspired Trajectories

A specialized package for creating high-quality 3D visualizations of spacecraft
docking trajectories, evolutionary algorithm results, and space environments
using Blender's Python API.

This package is designed to work alongside the main bioinspired package,
providing advanced 3D rendering capabilities for automated spacecraft docking simulations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main classes for easy access
from .core import BlenderScene, MaterialManager
from .trajectory import TrajectoryRenderer, AnimationManager
from .spacecraft import SpacecraftRenderer, DockingPortRenderer
from .environments import SpaceEnvironmentRenderer
# from .exporters import VideoExporter, ImageExporter

__all__ = [
    "BlenderScene",
    "MaterialManager", 
    "TrajectoryRenderer",
    "AnimationManager",
    "SpacecraftRenderer",
    "DockingPortRenderer", 
    "SpaceEnvironmentRenderer",
    # "VideoExporter",
    # "ImageExporter",
    "__version__"
]

# Check if Blender is available
try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    import warnings
    warnings.warn(
        "Blender Python API (bpy) not available. "
        "This package requires running inside Blender or with Blender's Python."
    )
