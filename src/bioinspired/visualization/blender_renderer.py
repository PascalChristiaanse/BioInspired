"""
Blender trajectory renderer for the BioInspired project.
This module provides tools to render trajectories and molecular docking results in Blender.
"""

import bpy
import bmesh
import numpy as np
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from mathutils import Vector, Matrix
import json

# Add the project src to path for importing bioinspired modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from bioinspired.data import DatabaseManager
    from bioinspired.data.models import Trajectory, Individual
except ImportError:
    print("Warning: Could not import bioinspired modules. Database features disabled.")
    DatabaseManager = None


class BlenderTrajectoryRenderer:
    """Render molecular docking trajectories in Blender."""
    
    def __init__(self, clear_scene: bool = True):
        """
        Initialize the Blender renderer.
        
        Args:
            clear_scene: Whether to clear the default scene
        """
        self.clear_scene = clear_scene
        if clear_scene:
            self.clear_default_scene()
        
        # Material library
        self.materials = {}
        self.setup_default_materials()
    
    def clear_default_scene(self):
        """Clear the default Blender scene."""
        # Delete default objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Remove default materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)
    
    def setup_default_materials(self):
        """Create default materials for different trajectory types."""
        # Trajectory path material
        self.materials['trajectory'] = self.create_material(
            name="TrajectoryPath",
            color=(0.1, 0.5, 1.0, 1.0),  # Blue
            metallic=0.3,
            roughness=0.4
        )
        
        # Start point material
        self.materials['start'] = self.create_material(
            name="StartPoint",
            color=(0.2, 1.0, 0.2, 1.0),  # Green
            metallic=0.0,
            roughness=0.2,
            emission_strength=0.5
        )
        
        # End point material
        self.materials['end'] = self.create_material(
            name="EndPoint",
            color=(1.0, 0.2, 0.2, 1.0),  # Red
            metallic=0.0,
            roughness=0.2,
            emission_strength=0.5
        )
        
        # Molecule material
        self.materials['molecule'] = self.create_material(
            name="Molecule",
            color=(0.8, 0.8, 0.8, 0.8),  # Light gray, semi-transparent
            metallic=0.1,
            roughness=0.6,
            alpha=0.7
        )
    
    def create_material(self, name: str, color: Tuple[float, float, float, float], 
                       metallic: float = 0.0, roughness: float = 0.5, 
                       emission_strength: float = 0.0, alpha: float = 1.0):
        """Create a material with specified properties."""
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        
        # Get the principled BSDF node
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs[0].default_value = color  # Base Color
        bsdf.inputs[1].default_value = metallic  # Metallic
        bsdf.inputs[2].default_value = roughness  # Roughness
        bsdf.inputs[21].default_value = alpha  # Alpha
        
        if emission_strength > 0:
            bsdf.inputs[20].default_value = emission_strength  # Emission Strength
            bsdf.inputs[19].default_value = color[:3] + (1.0,)  # Emission Color
        
        if alpha < 1.0:
            mat.blend_method = 'BLEND'
        
        return mat
    
    def render_trajectory_path(self, trajectory: np.ndarray, name: str = "Trajectory", 
                             curve_resolution: int = 12, bevel_depth: float = 0.02) -> bpy.types.Object:
        """
        Render a trajectory as a 3D curve.
        
        Args:
            trajectory: Nx3 array of trajectory points
            name: Name for the trajectory object
            curve_resolution: Resolution of the curve
            bevel_depth: Thickness of the trajectory line
        
        Returns:
            The created Blender curve object
        """
        # Create curve data
        curve_data = bpy.data.curves.new(name=f"{name}_curve", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = curve_resolution
        curve_data.bevel_depth = bevel_depth
        
        # Create spline
        spline = curve_data.splines.new('NURBS')
        spline.points.add(len(trajectory) - 1)  # -1 because spline starts with 1 point
        
        # Set points
        for i, point in enumerate(trajectory):
            spline.points[i].co = (point[0], point[1], point[2], 1.0)
        
        # Create object
        curve_obj = bpy.data.objects.new(name, curve_data)
        bpy.context.collection.objects.link(curve_obj)
        
        # Assign material
        if 'trajectory' in self.materials:
            curve_obj.data.materials.append(self.materials['trajectory'])
        
        return curve_obj
    
    def render_trajectory_spheres(self, trajectory: np.ndarray, name: str = "TrajectoryPoints",
                                 sphere_size: float = 0.05, subsample: int = 5) -> List[bpy.types.Object]:
        """
        Render trajectory points as spheres.
        
        Args:
            trajectory: Nx3 array of trajectory points
            name: Base name for sphere objects
            sphere_size: Radius of each sphere
            subsample: Only render every nth point
        
        Returns:
            List of created sphere objects
        """
        spheres = []
        
        # Subsample trajectory for performance
        sampled_trajectory = trajectory[::subsample]
        
        for i, point in enumerate(sampled_trajectory):
            # Create sphere
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=sphere_size,
                location=(point[0], point[1], point[2])
            )
            
            sphere = bpy.context.active_object
            sphere.name = f"{name}_point_{i}"
            
            # Assign material based on position
            if i == 0 and 'start' in self.materials:
                sphere.data.materials.append(self.materials['start'])
            elif i == len(sampled_trajectory) - 1 and 'end' in self.materials:
                sphere.data.materials.append(self.materials['end'])
            elif 'trajectory' in self.materials:
                sphere.data.materials.append(self.materials['trajectory'])
            
            spheres.append(sphere)
        
        return spheres
    
    def render_molecule_placeholder(self, position: Tuple[float, float, float] = (0, 0, 0),
                                  scale: float = 1.0, name: str = "Molecule") -> bpy.types.Object:
        """
        Render a placeholder molecule (as a complex mesh).
        
        Args:
            position: Position of the molecule
            scale: Scale factor
            name: Name of the molecule object
        
        Returns:
            The created molecule object
        """
        # Create a complex shape to represent a molecule
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=2,
            location=position,
            scale=(scale, scale, scale)
        )
        
        molecule = bpy.context.active_object
        molecule.name = name
        
        # Add some complexity with modifiers
        # Displacement modifier for surface roughness
        disp_mod = molecule.modifiers.new(name="Displacement", type='DISPLACE')
        disp_mod.strength = 0.1
        
        # Assign material
        if 'molecule' in self.materials:
            molecule.data.materials.append(self.materials['molecule'])
        
        return molecule
    
    def setup_lighting(self, lighting_type: str = "three_point"):
        """
        Set up lighting for the scene.
        
        Args:
            lighting_type: Type of lighting setup ('three_point', 'hdri', 'simple')
        """
        if lighting_type == "three_point":
            # Key light
            bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
            key_light = bpy.context.active_object
            key_light.name = "KeyLight"
            key_light.data.energy = 3.0
            
            # Fill light
            bpy.ops.object.light_add(type='AREA', location=(-3, 2, 5))
            fill_light = bpy.context.active_object
            fill_light.name = "FillLight"
            fill_light.data.energy = 1.5
            fill_light.data.size = 2.0
            
            # Rim light
            bpy.ops.object.light_add(type='SPOT', location=(0, -5, 3))
            rim_light = bpy.context.active_object
            rim_light.name = "RimLight"
            rim_light.data.energy = 2.0
            rim_light.data.spot_size = 1.2
        
        elif lighting_type == "simple":
            # Single sun light
            bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
            sun_light = bpy.context.active_object
            sun_light.name = "SunLight"
            sun_light.data.energy = 5.0
    
    def setup_camera(self, target_location: Tuple[float, float, float] = (0, 0, 0),
                    camera_location: Tuple[float, float, float] = (7, -7, 5)):
        """
        Set up and position the camera.
        
        Args:
            target_location: Point for camera to look at
            camera_location: Position of the camera
        """
        # Add camera
        bpy.ops.object.camera_add(location=camera_location)
        camera = bpy.context.active_object
        camera.name = "MainCamera"
        
        # Point camera at target
        direction = Vector(target_location) - Vector(camera_location)
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        
        # Set as active camera
        bpy.context.scene.camera = camera
        
        return camera
    
    def render_scene(self, output_path: str, resolution: Tuple[int, int] = (1920, 1080),
                    samples: int = 128, engine: str = 'CYCLES'):
        """
        Render the current scene.
        
        Args:
            output_path: Path to save the rendered image
            resolution: Resolution (width, height)
            samples: Number of samples for rendering
            engine: Render engine ('CYCLES' or 'EEVEE')
        """
        # Set render settings
        scene = bpy.context.scene
        scene.render.engine = engine
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        scene.render.filepath = output_path
        
        if engine == 'CYCLES':
            scene.cycles.samples = samples
        elif engine == 'EEVEE':
            scene.eevee.taa_render_samples = samples
        
        # Render
        bpy.ops.render.render(write_still=True)
    
    def create_animation(self, trajectory: np.ndarray, frame_count: int = 250,
                        camera_follows: bool = True) -> None:
        """
        Create an animation following the trajectory.
        
        Args:
            trajectory: Nx3 array of trajectory points
            frame_count: Number of animation frames
            camera_follows: Whether camera should follow the trajectory
        """
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = frame_count
        
        if camera_follows and scene.camera:
            # Animate camera to follow trajectory
            camera = scene.camera
            
            # Clear existing keyframes
            camera.animation_data_clear()
            
            for frame in range(frame_count):
                # Interpolate position along trajectory
                t = frame / (frame_count - 1)
                idx = int(t * (len(trajectory) - 1))
                idx = min(idx, len(trajectory) - 1)
                
                # Set camera position
                camera.location = Vector(trajectory[idx]) + Vector((3, -3, 2))
                camera.keyframe_insert(data_path="location", frame=frame + 1)
                
                # Point camera at trajectory point
                if idx < len(trajectory) - 1:
                    look_at = Vector(trajectory[idx])
                    direction = look_at - camera.location
                    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
                    camera.keyframe_insert(data_path="rotation_euler", frame=frame + 1)


class TrajectoryLoader:
    """Load trajectories from the BioInspired database for Blender rendering."""
    
    def __init__(self):
        """Initialize the trajectory loader."""
        if DatabaseManager is None:
            raise ImportError("DatabaseManager not available. Check your Python path.")
        
        self.db = DatabaseManager()
    
    def load_trajectory_by_id(self, trajectory_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a trajectory from the database by ID.
        
        Args:
            trajectory_id: Database ID of the trajectory
        
        Returns:
            Tuple of (trajectory_data, metadata)
        """
        trajectory_obj, trajectory_data = self.db.load_trajectory_data(trajectory_id)
        
        metadata = {
            'id': trajectory_obj.id,
            'individual_id': trajectory_obj.individual_id,
            'steps': trajectory_obj.steps,
            'format': trajectory_obj.format,
            'metadata': trajectory_obj.trajectory_metadata
        }
        
        return trajectory_data, metadata
    
    def load_best_trajectories(self, algorithm_id: int, limit: int = 5) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Load trajectories for the best individuals from an algorithm run.
        
        Args:
            algorithm_id: Database ID of the algorithm run
            limit: Maximum number of trajectories to load
        
        Returns:
            List of (trajectory_data, metadata) tuples
        """
        # Get best individuals
        best_individuals = self.db.get_best_individuals(algorithm_id, limit=limit)
        
        trajectories = []
        for individual in best_individuals:
            # Find trajectory for this individual
            from bioinspired.data.database import get_session
            from bioinspired.data.models import Trajectory
            
            with get_session() as session:
                trajectory_record = session.query(Trajectory).filter_by(
                    individual_id=individual.id
                ).first()
                
                if trajectory_record:
                    trajectory_data, metadata = self.load_trajectory_by_id(trajectory_record.id)
                    metadata['fitness'] = individual.fitness
                    metadata['generation'] = individual.generation
                    metadata['species'] = individual.species
                    trajectories.append((trajectory_data, metadata))
        
        return trajectories


# Blender operator for easy use in Blender UI
class BIOINSPIRED_OT_render_trajectory(bpy.types.Operator):
    """Render BioInspired Trajectory"""
    bl_idname = "bioinspired.render_trajectory"
    bl_label = "Render Trajectory"
    bl_options = {'REGISTER', 'UNDO'}
    
    trajectory_id: bpy.props.IntProperty(
        name="Trajectory ID",
        description="Database ID of the trajectory to render",
        default=1,
        min=1
    )
    
    def execute(self, context):
        try:
            loader = TrajectoryLoader()
            renderer = BlenderTrajectoryRenderer()
            
            # Load trajectory
            trajectory_data, metadata = loader.load_trajectory_by_id(self.trajectory_id)
            
            # Render trajectory
            renderer.render_trajectory_path(trajectory_data, f"Trajectory_{self.trajectory_id}")
            renderer.render_trajectory_spheres(trajectory_data, f"Points_{self.trajectory_id}")
            
            # Setup scene
            renderer.setup_lighting()
            renderer.setup_camera()
            
            self.report({'INFO'}, f"Rendered trajectory {self.trajectory_id} with {len(trajectory_data)} points")
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to render trajectory: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


# Register the operator
def register():
    bpy.utils.register_class(BIOINSPIRED_OT_render_trajectory)

def unregister():
    bpy.utils.unregister_class(BIOINSPIRED_OT_render_trajectory)

if __name__ == "__main__":
    register()
