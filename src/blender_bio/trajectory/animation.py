"""
Animation and keyframe management.
"""

import numpy as np
from typing import Tuple, Optional, Any
import json
import ast

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


class AnimationManager:
    """Manages animation and keyframes for trajectory visualization."""
    
    def __init__(self, scale_factor: float = 1.0 / 1000000.0):
        """Initialize animation manager."""
        if not BLENDER_AVAILABLE:
            raise ImportError("Blender Python API not available")
        
        self.animated_objects = []
        self.scale_factor = scale_factor
        
    def create_spacecraft_object(self, name: str = "spacecraft", size: float = 1.0) -> Any:
        """Create a simple spacecraft object for animation."""
        # Create a simple spacecraft representation using a cube or arrow
        bpy.ops.mesh.primitive_cube_add(size=size)
        spacecraft = bpy.context.object
        spacecraft.name = name
        
        # Add a material
        material = bpy.data.materials.new(name=f"{name}_material")
        material.use_nodes = True
        material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.3, 0.1, 1.0)  # Orange color
        spacecraft.data.materials.append(material)
        
        self.animated_objects.append(spacecraft)
        return spacecraft
    
    def extract_trajectory_data(self, trajectory_record) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract position, velocity, and time data from trajectory record."""
        try:
            sim_data = trajectory_record.dynamics_simulator
            
            # Handle different data formats
            if sim_data is None or (hasattr(sim_data, '__len__') and len(sim_data) == 0):
                return None, None, None
            
            # Normalize sim_data to a dict
            sim_dict = None
            if isinstance(sim_data, dict):
                sim_dict = sim_data
            elif isinstance(sim_data, (list, tuple)) and len(sim_data) > 0:
                if isinstance(sim_data[0], dict):
                    sim_dict = sim_data[0]
                else:
                    sim_data_str = sim_data[0]
                    try:
                        sim_dict = json.loads(sim_data_str)
                    except Exception:
                        sim_dict = ast.literal_eval(sim_data_str)
            elif isinstance(sim_data, str):
                try:
                    sim_dict = json.loads(sim_data)
                except Exception:
                    sim_dict = ast.literal_eval(sim_data)
            
            if not sim_dict or 'state_history' not in sim_dict:
                return None, None, None
            
            state_history = sim_dict['state_history']
            if not isinstance(state_history, dict) or len(state_history) == 0:
                return None, None, None
            
            # Extract data points
            sorted_times = sorted(state_history.keys(), key=float)
            times = np.array([float(t) for t in sorted_times])
            positions = np.array([state_history[t][:3] for t in sorted_times])
            velocities = np.array([state_history[t][3:6] for t in sorted_times]) if len(state_history[sorted_times[0]]) >= 6 else None
            
            return positions, velocities, times
            
        except Exception as e:
            print(f"Error extracting trajectory data: {e}")
            return None, None, None
    
    def animate_object_along_trajectory(self, obj: Any, positions: np.ndarray, 
                                      velocities: Optional[np.ndarray] = None,
                                      times: Optional[np.ndarray] = None,
                                      total_frames: int = 250,
                                      frame_start: int = 1) -> bool:
        """Animate an object along a trajectory using keyframes."""
        
        if positions is None or len(positions) == 0:
            print("No position data available for animation")
            return False
        
        # Scale positions
        scaled_positions = positions * self.scale_factor
        
        # Clear existing animation data
        obj.animation_data_clear()
        
        # Set up time mapping
        if times is not None:
            time_duration = times[-1] - times[0]
            if time_duration > 0:
                # Map simulation time to frame numbers
                frame_times = ((times - times[0]) / time_duration) * (total_frames - 1) + frame_start
            else:
                frame_times = np.linspace(frame_start, frame_start + total_frames - 1, len(positions))
        else:
            frame_times = np.linspace(frame_start, frame_start + total_frames - 1, len(positions))
        
        # Set keyframes for position
        for i, (pos, frame_time) in enumerate(zip(scaled_positions, frame_times)):
            obj.location = pos
            obj.keyframe_insert(data_path="location", frame=int(frame_time))
        
        # Set keyframes for rotation based on velocity (if available)
        if velocities is not None:
            for i, (vel, frame_time) in enumerate(zip(velocities, frame_times)):
                if np.linalg.norm(vel) > 1e-6:  # Avoid division by zero
                    # Calculate rotation to align with velocity direction
                    vel_normalized = vel / np.linalg.norm(vel)
                    
                    # Create rotation matrix to align object with velocity
                    # Default forward direction is +Y in Blender
                    default_forward = np.array([0, 1, 0])
                    
                    # Calculate rotation axis and angle
                    axis = np.cross(default_forward, vel_normalized)
                    axis_length = np.linalg.norm(axis)
                    
                    if axis_length > 1e-6:
                        axis = axis / axis_length
                        angle = np.arccos(np.clip(np.dot(default_forward, vel_normalized), -1, 1))
                        
                        # Convert to quaternion
                        try:
                            import mathutils
                            rotation_quat = mathutils.Quaternion(axis, angle)
                            obj.rotation_quaternion = rotation_quat
                            obj.keyframe_insert(data_path="rotation_quaternion", frame=int(frame_time))
                        except ImportError:
                            # Fallback if mathutils not available
                            pass
        
        # Set interpolation mode to linear for smooth animation
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'LINEAR'
        
        return True
    
    def animate_trajectory_growth(self, trajectory_obj, frames: int = 100):
        """Animate trajectory appearing over time using build modifier."""
        try:
            # Add build modifier to make trajectory appear over time
            build_modifier = trajectory_obj.modifiers.new(name="Build", type='BUILD')
            build_modifier.frame_start = 1
            build_modifier.frame_duration = frames
            build_modifier.use_random_order = False
            
            return True
        except Exception as e:
            print(f"Error animating trajectory growth: {e}")
            return False
    
    def animate_camera_path(self, trajectory_data: np.ndarray, frames: int = 200, 
                           camera_offset: np.ndarray = np.array([0, -10, 5]),
                           target_offset: np.ndarray = np.array([0, 0, 0])):
        """Animate camera following the trajectory."""
        try:
            # Get or create camera
            if 'Camera' in bpy.data.objects:
                camera = bpy.data.objects['Camera']
            else:
                bpy.ops.object.camera_add()
                camera = bpy.context.object
                camera.name = 'Camera'
            
            # Scale trajectory data
            scaled_trajectory = trajectory_data * self.scale_factor
            
            # Clear existing animation
            camera.animation_data_clear()
            
            # Create keyframes for camera position and rotation
            for i, frame in enumerate(np.linspace(1, frames, len(scaled_trajectory))):
                pos = scaled_trajectory[i]
                
                # Set camera position with offset
                camera.location = pos + camera_offset * self.scale_factor
                camera.keyframe_insert(data_path="location", frame=int(frame))
                
                # Make camera look at the trajectory point
                look_at = pos + target_offset * self.scale_factor
                direction = look_at - camera.location
                
                # Calculate rotation to look at target
                try:
                    import mathutils
                    rot_quat = direction.to_track_quat('-Z', 'Y')
                    camera.rotation_quaternion = rot_quat
                    camera.keyframe_insert(data_path="rotation_quaternion", frame=int(frame))
                except ImportError:
                    # Fallback if mathutils not available
                    pass
            
            # Set interpolation mode
            if camera.animation_data and camera.animation_data.action:
                for fcurve in camera.animation_data.action.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        keyframe.interpolation = 'LINEAR'
            
            return True
        except Exception as e:
            print(f"Error animating camera: {e}")
            return False
    
    def set_animation_range(self, start_frame: int = 1, end_frame: int = 250):
        """Set the animation frame range."""
        bpy.context.scene.frame_start = start_frame
        bpy.context.scene.frame_end = end_frame
        bpy.context.scene.frame_current = start_frame
