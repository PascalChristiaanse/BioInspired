"""
Example: Using the blender_bio module to render a simple trajectory in Blender

This script demonstrates how to use the blender_bio package to render a trajectory
in Blender from Python. Run this script from Blender's scripting workspace or using
Blender's command line with the --python flag.
"""

import sys
import os
import numpy as np

# Ensure the src directory is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import the BlenderBio path renderer
from blender_bio.trajectory.path_renderer import PathRenderer

# Example trajectory: a simple helix
num_points = 100
theta = np.linspace(0, 4 * np.pi, num_points)
z = np.linspace(0, 2, num_points)
x = np.cos(theta)
y = np.sin(theta)
trajectory = np.stack((x, y, z), axis=1)

# Create a path renderer and render the trajectory
renderer = PathRenderer()
renderer.render_path(trajectory, name="ExampleHelix", color=(0.2, 0.7, 1.0, 1.0), thickness=0.05)

print("Trajectory rendered in Blender using blender_bio!")
