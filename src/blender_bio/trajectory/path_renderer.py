"""
Trajectory path rendering and visualization.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

# Add the main project to path for importing bioinspired
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import bpy
    import bmesh
    from mathutils import Vector, Color
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


class TrajectoryRenderer:
    """Renders molecular docking trajectories in Blender."""
    
    def __init__(self, scene_manager=None):
        """Initialize trajectory renderer."""
        if not BLENDER_AVAILABLE:
            raise ImportError("Blender Python API not available")
        
        self.scene_manager = scene_manager
        self.trajectory_objects = []
        
    def load_trajectory_from_database(self, trajectory_id: int):
        """Load trajectory data from the database."""
        try:
            from bioinspired.data import DatabaseManager
            db = DatabaseManager()
            trajectory_obj, trajectory_data = db.load_trajectory_data(trajectory_id)
            return trajectory_data, trajectory_obj
        except ImportError:
            raise ImportError("Cannot import bioinspired package. Make sure it's in the Python path.")
    
    def render_trajectory_path(self, trajectory_data: np.ndarray, 
                             name: str = "trajectory",
                             path_width: float = 0.05,
                             color: Tuple[float, float, float, float] = (0.2, 0.8, 0.3, 1.0)):
        """Render a trajectory as a 3D path."""
        
        # Create curve from trajectory data
        curve_data = bpy.data.curves.new(f"{name}_curve", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 12
        
        # Create spline
        polyline = curve_data.splines.new('NURBS')
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
        material = self.create_trajectory_material(f"{name}_material", color)
        curve_obj.data.materials.append(material)
        
        self.trajectory_objects.append(curve_obj)
        return curve_obj
    
    def render_trajectory_spheres(self, trajectory_data: np.ndarray,
                                name: str = "trajectory_spheres",
                                sphere_size: float = 0.1,
                                color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
                                subsample: int = 5):
        """Render trajectory points as spheres."""
        
        # Subsample trajectory for performance
        points = trajectory_data[::subsample]
        
        # Create material
        material = self.create_trajectory_material(f"{name}_material", color)
        
        sphere_objects = []
        for i, point in enumerate(points):
            # Create sphere
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=sphere_size,
                location=(point[0], point[1], point[2])
            )
            sphere = bpy.context.object
            sphere.name = f"{name}_sphere_{i}"
            
            # Assign material
            sphere.data.materials.append(material)
            sphere_objects.append(sphere)
        
        self.trajectory_objects.extend(sphere_objects)
        return sphere_objects
    
    def render_start_end_markers(self, trajectory_data: np.ndarray,
                                name: str = "markers",
                                start_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                                end_color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
                                marker_size: float = 0.3):
        """Add start and end markers to trajectory."""
        
        start_point = trajectory_data[0]
        end_point = trajectory_data[-1]
        
        # Start marker (green sphere)
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=marker_size,
            location=(start_point[0], start_point[1], start_point[2])
        )
        start_marker = bpy.context.object
        start_marker.name = f"{name}_start"
        
        start_material = self.create_trajectory_material(f"{name}_start_material", start_color)
        start_marker.data.materials.append(start_material)
        
        # End marker (red cube)
        bpy.ops.mesh.primitive_cube_add(
            size=marker_size * 2,
            location=(end_point[0], end_point[1], end_point[2])
        )
        end_marker = bpy.context.object
        end_marker.name = f"{name}_end"
        
        end_material = self.create_trajectory_material(f"{name}_end_material", end_color)
        end_marker.data.materials.append(end_material)
        
        self.trajectory_objects.extend([start_marker, end_marker])
        return start_marker, end_marker
    
    def create_trajectory_material(self, name: str, 
                                 color: Tuple[float, float, float, float],
                                 metallic: float = 0.0,
                                 roughness: float = 0.4) -> bpy.types.Material:
        """Create a material for trajectory visualization."""
        
        # Create material
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        
        # Get principled BSDF node
        bsdf = material.node_tree.nodes["Principled BSDF"]
        
        # Set properties
        bsdf.inputs[0].default_value = color  # Base Color
        bsdf.inputs[6].default_value = metallic  # Metallic
        bsdf.inputs[9].default_value = roughness  # Roughness
        
        return material
    
    def render_multiple_trajectories(self, trajectories: List[np.ndarray],
                                   names: Optional[List[str]] = None,
                                   colors: Optional[List[Tuple[float, float, float, float]]] = None):
        """Render multiple trajectories with different colors."""
        
        if names is None:
            names = [f"trajectory_{i}" for i in range(len(trajectories))]
        
        if colors is None:
            # Generate distinct colors
            colors = self.generate_distinct_colors(len(trajectories))
        
        rendered_objects = []
        for i, (trajectory, name, color) in enumerate(zip(trajectories, names, colors)):
            obj = self.render_trajectory_path(trajectory, name, color=color)
            rendered_objects.append(obj)
        
        return rendered_objects
    
    def generate_distinct_colors(self, num_colors: int) -> List[Tuple[float, float, float, float]]:
        """Generate visually distinct colors for multiple trajectories."""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            # Convert HSV to RGB (simplified)
            if hue < 1/6:
                rgb = (1, 6*hue, 0)
            elif hue < 2/6:
                rgb = (2-6*hue, 1, 0)
            elif hue < 3/6:
                rgb = (0, 1, 6*hue-2)
            elif hue < 4/6:
                rgb = (0, 4-6*hue, 1)
            elif hue < 5/6:
                rgb = (6*hue-4, 0, 1)
            else:
                rgb = (1, 0, 6-6*hue)
            
            colors.append((*rgb, 1.0))
        
        return colors
    
    def clear_trajectory_objects(self):
        """Remove all trajectory objects from the scene."""
        for obj in self.trajectory_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        self.trajectory_objects.clear()
