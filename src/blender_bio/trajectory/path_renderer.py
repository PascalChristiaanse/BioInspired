"""
Trajectory path rendering and visualization for the BioInspired project.
This module connects to the project's database to fetch trajectory data
and renders it within Blender.
"""

import sys
import os
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional

# Add project paths to Blender's Python environment
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import bpy
    from src.bioinspired.data.services import get_trajectory
    from src.bioinspired.data.models import Trajectory
except ImportError as e:
    raise ImportError(
        "This script must be run within Blender's Python environment, "
        f"and the 'bioinspired' package must be accessible. Error: {e}"
    )


class TrajectoryRenderer:
    """Renders simulation trajectories in Blender."""

    def __init__(self, scene_manager=None, scale_factor: float = 1.0 / 1000000.0):
        """
        Initialize trajectory renderer.

        Args:
            scene_manager: The scene manager for handling Blender scene operations.
            scale_factor: Factor to scale down simulation coordinates (e.g., 1e-3 for m to km).
        """
        self.scene_manager = scene_manager
        self.scale_factor = scale_factor
        self.trajectory_objects = []

    def load_trajectory_from_database(self, trajectory_id: int) -> Tuple[Optional[np.ndarray], Optional[Trajectory]]:
        """
        Load trajectory data from the database using the new services.
        
        Args:
            trajectory_id: The database ID of the trajectory to load.
            
        Returns:
            A tuple containing:
            - A numpy array of shape (N, 3) with trajectory coordinates.
            - The SQLAlchemy Trajectory object.
            Returns (None, None) if loading fails.
        """
        print(f"Loading trajectory with ID: {trajectory_id}")
        try:
            trajectory_record = get_trajectory(trajectory_id)
            if not trajectory_record:
                print(f"Error: Trajectory with ID {trajectory_id} not found.")
                return None, None

            # The dynamics_simulator field is expected to be a dict/JSON
            sim_data = trajectory_record.dynamics_simulator
            if isinstance(sim_data[0], str):
                sim_data = json.loads(sim_data[0])

            if not sim_data or 'state_history' not in sim_data:
                print(f"Error: No state history found in trajectory {trajectory_id}.")
                return None, None

            state_history = sim_data['state_history']
            
            # Sort state history by time (keys are strings of floats)
            # and extract position (first 3 elements of each state vector)
            sorted_times = sorted(state_history.keys(), key=float)
            trajectory_points = np.array([state_history[t][:3] for t in sorted_times])
            
            print(f"Successfully loaded {len(trajectory_points)} points for trajectory {trajectory_id}.")
            return trajectory_points, trajectory_record
            
        except Exception as e:
            print(f"An unexpected error occurred while loading trajectory {trajectory_id}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def render_trajectory_path(self, trajectory_data: np.ndarray,
                             name: str = "trajectory",
                             path_width: float = 0.05,
                             color: Tuple[float, float, float, float] = (0.2, 0.8, 0.3, 1.0)):
        """Render a trajectory as a 3D path."""
        
        # Scale trajectory data
        trajectory_data = trajectory_data * self.scale_factor
        
        # Create curve from trajectory data
        curve_data = bpy.data.curves.new(f"{name}_curve", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 2 # Lower resolution for performance
        
        # Create spline
        polyline = curve_data.splines.new('POLY') # POLY is better for raw data
        polyline.points.add(len(trajectory_data) - 1)
        
        # Add points
        for i, point in enumerate(trajectory_data):
            polyline.points[i].co = (point[0], point[1], point[2], 1)
        
        # Set curve properties
        curve_data.bevel_depth = path_width
        curve_data.bevel_resolution = 4
        curve_data.fill_mode = 'FULL'
        
        # Create object
        curve_obj = bpy.data.objects.new(f"{name}_path", curve_data)
        bpy.context.scene.collection.objects.link(curve_obj)
        
        # Create and assign material
        material = self.create_material(f"{name}_material", color)
        curve_obj.data.materials.append(material)
        
        self.trajectory_objects.append(curve_obj)
        return curve_obj

    def render_trajectory_spheres(self, trajectory_data: np.ndarray,
                                name: str = "trajectory_spheres",
                                sphere_size: float = 0.1,
                                color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
                                subsample: int = 1):
        """Render trajectory points as spheres."""
        
        # Scale and subsample trajectory for performance
        trajectory_data = trajectory_data * self.scale_factor
        points = trajectory_data[::subsample]
        
        # Create material
        material = self.create_material(f"{name}_material", color)
        
        sphere_objects = []
        for i, point in enumerate(points):
            # Create sphere
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=sphere_size,
                location=(point[0], point[1], point[2]),
                segments=16, # Lower resolution for performance
                ring_count=8
            )
            sphere = bpy.context.object
            sphere.name = f"{name}_sphere_{i}"
            
            # Assign material and smooth shading
            sphere.data.materials.append(material)
            bpy.ops.object.shade_smooth()
            sphere_objects.append(sphere)
        
        self.trajectory_objects.extend(sphere_objects)
        return sphere_objects

    def render_start_end_markers(self, trajectory_data: np.ndarray,
                                name: str = "markers",
                                start_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                                end_color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
                                marker_size: float = 0.3):
        """Add start and end markers to trajectory."""
        
        if len(trajectory_data) == 0:
            return None, None

        # Scale trajectory data
        trajectory_data = trajectory_data * self.scale_factor

        start_point = trajectory_data[0]
        end_point = trajectory_data[-1]
        
        # Start marker (green sphere)
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=marker_size,
            location=(start_point[0], start_point[1], start_point[2])
        )
        start_marker = bpy.context.object
        start_marker.name = f"{name}_start"
        
        start_material = self.create_material(f"{name}_start_material", start_color)
        start_marker.data.materials.append(start_material)
        
        # End marker (red cube)
        bpy.ops.mesh.primitive_cube_add(
            size=marker_size * 2,
            location=(end_point[0], end_point[1], end_point[2])
        )
        end_marker = bpy.context.object
        end_marker.name = f"{name}_end"
        
        end_material = self.create_material(f"{name}_end_material", end_color)
        end_marker.data.materials.append(end_material)
        
        self.trajectory_objects.extend([start_marker, end_marker])
        return start_marker, end_marker

    def create_material(self, name: str,
                      color: Tuple[float, float, float, float],
                      metallic: float = 0.1,
                      roughness: float = 0.5) -> bpy.types.Material:
        """Create a material for trajectory visualization."""
        
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Metallic'].default_value = metallic
            bsdf.inputs['Roughness'].default_value = roughness
        
        return material

    def render_multiple_trajectories(self, trajectories: List[np.ndarray],
                                   names: Optional[List[str]] = None,
                                   colors: Optional[List[Tuple[float, float, float, float]]] = None):
        """Render multiple trajectories with different colors."""
        
        if names is None:
            names = [f"trajectory_{i}" for i in range(len(trajectories))]
        
        if colors is None:
            colors = self.generate_distinct_colors(len(trajectories))
        
        rendered_objects = []
        for i, (trajectory, name, color) in enumerate(zip(trajectories, names, colors)):
            obj = self.render_trajectory_path(trajectory, name, color=color)
            rendered_objects.append(obj)
        
        return rendered_objects

    def generate_distinct_colors(self, num_colors: int) -> List[Tuple[float, float, float, float]]:
        """Generate visually distinct colors for multiple trajectories."""
        import colorsys
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append((*rgb, 1.0))
        return colors

    def clear_trajectory_objects(self):
        """Remove all trajectory objects from the scene."""
        for obj in self.trajectory_objects:
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        self.trajectory_objects.clear()
