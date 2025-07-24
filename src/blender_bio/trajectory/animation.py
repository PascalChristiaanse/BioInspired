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
    
    def extract_trajectory_data(self, trajectory_record) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract position, velocity, time, quaternion, and angular velocity data from trajectory record.
        
        Returns:
            Tuple of (positions, velocities, times, quaternions, angular_velocities)
            - positions: (N, 3) array of x, y, z positions
            - velocities: (N, 3) array of vx, vy, vz velocities  
            - times: (N,) array of simulation times
            - quaternions: (N, 4) array of q0, q1, q2, q3 quaternions (if available)
            - angular_velocities: (N, 3) array of ωx, ωy, ωz angular velocities (if available)
        """
        try:
            sim_data = trajectory_record.dynamics_simulator
            
            # Handle different data formats
            if sim_data is None or (hasattr(sim_data, '__len__') and len(sim_data) == 0):
                return None, None, None, None, None
            
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
                return None, None, None, None, None
            
            state_history = sim_dict['state_history']
            if not isinstance(state_history, dict) or len(state_history) == 0:
                return None, None, None, None, None
            
            # Extract data points
            sorted_times = sorted(state_history.keys(), key=float)
            times = np.array([float(t) for t in sorted_times])
            
            # Get state vector length to determine what data is available
            first_state = state_history[sorted_times[0]]
            state_length = len(first_state)
            
            # Always extract positions (first 3 elements)
            positions = np.array([state_history[t][:3] for t in sorted_times])
            
            # Extract velocities if available (elements 3-6)
            velocities = None
            if state_length >= 6:
                velocities = np.array([state_history[t][3:6] for t in sorted_times])
            
            # Extract quaternions if available (elements 6-10 for rotational spacecraft)
            quaternions = None
            if state_length >= 10:
                quaternions = np.array([state_history[t][6:10] for t in sorted_times])
            
            # Extract angular velocities if available (elements 10-13)
            angular_velocities = None
            if state_length >= 13:
                angular_velocities = np.array([state_history[t][10:13] for t in sorted_times])
            
            print(f"Extracted trajectory data: positions={positions.shape}, velocities={velocities.shape if velocities is not None else None}, quaternions={quaternions.shape if quaternions is not None else None}, angular_velocities={angular_velocities.shape if angular_velocities is not None else None}")
            
            return positions, velocities, times, quaternions, angular_velocities
            
        except Exception as e:
            print(f"Error extracting trajectory data: {e}")
            return None, None, None, None, None
    
    def animate_object_along_trajectory(self, obj: Any, positions: np.ndarray, 
                                      velocities: Optional[np.ndarray] = None,
                                      times: Optional[np.ndarray] = None,
                                      quaternions: Optional[np.ndarray] = None,
                                      angular_velocities: Optional[np.ndarray] = None,
                                      total_frames: int = 250,
                                      frame_start: int = 1) -> bool:
        """Animate an object along a trajectory using keyframes.
        
        Args:
            obj: Blender object to animate
            positions: (N, 3) array of x, y, z positions
            velocities: (N, 3) array of velocities (optional, used for orientation if no quaternions)
            times: (N,) array of simulation times (optional)
            quaternions: (N, 4) array of quaternions for orientation (optional, preferred over velocities)
            angular_velocities: (N, 3) array of angular velocities (optional, currently not used)
            total_frames: Total number of animation frames
            frame_start: Starting frame number
            
        Returns:
            True if animation was successful, False otherwise
        """
        
        if positions is None or len(positions) == 0:
            print("No position data available for animation")
            return False
        
        # Analyze timing distribution for debugging
        timing_analysis = self.analyze_timing_distribution(times, total_frames)
        print(f"Timing analysis: {timing_analysis['message']}")
        
        # Scale positions
        scaled_positions = positions * self.scale_factor
        
        # Clear existing animation data
        obj.animation_data_clear()
        
        # Set rotation mode to quaternion for proper quaternion animation
        obj.rotation_mode = 'QUATERNION'
        
        # Set up time mapping for variable step sizes
        if times is not None:
            time_duration = times[-1] - times[0]
            if time_duration > 0:
                # Map simulation time to frame numbers based on actual time progression
                frame_times = ((times - times[0]) / time_duration) * (total_frames - 1) + frame_start
                print(f"Using variable time steps: {len(times)} time points mapped to {total_frames} frames")
                print(f"Time range: {times[0]:.3f}s to {times[-1]:.3f}s (duration: {time_duration:.3f}s)")
            else:
                print("Warning: Zero time duration detected, using uniform frame spacing")
                frame_times = np.linspace(frame_start, frame_start + total_frames - 1, len(positions))
        else:
            print("Using uniform frame spacing (no time data provided)")
            frame_times = np.linspace(frame_start, frame_start + total_frames - 1, len(positions))
        
        # Set keyframes for position
        for i, (pos, frame_time) in enumerate(zip(scaled_positions, frame_times)):
            obj.location = pos
            obj.keyframe_insert(data_path="location", frame=int(frame_time))
        
        # Set keyframes for rotation
        if quaternions is not None:
            # Use actual quaternion data from rotational dynamics (preferred method)
            print("Animating spacecraft orientation using quaternion data")
            for i, (quat, frame_time) in enumerate(zip(quaternions, frame_times)):
                try:
                    import mathutils
                    # Convert quaternion (w, x, y, z) format to Blender format
                    # Tudat quaternions are typically in (w, x, y, z) format
                    # Blender quaternions are in (w, x, y, z) format as well
                    rotation_quat = mathutils.Quaternion((quat[0], quat[1], quat[2], quat[3]))
                    rotation_quat.normalize()
                    obj.rotation_quaternion = rotation_quat
                    obj.keyframe_insert(data_path="rotation_quaternion", frame=int(frame_time))
                except ImportError:
                    # Fallback if mathutils not available - use direct assignment
                    obj.rotation_quaternion = (quat[0], quat[1], quat[2], quat[3])
                    obj.keyframe_insert(data_path="rotation_quaternion", frame=int(frame_time))
                    
        elif velocities is not None:
            # Fallback: use velocity direction for orientation (less accurate but still useful)
            print("Animating spacecraft orientation using velocity direction")
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
        # For variable time steps, we might want to use BEZIER interpolation for smoother results
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    if times is not None:
                        # Use BEZIER interpolation for variable time steps for smoother animation
                        keyframe.interpolation = 'BEZIER'
                        keyframe.handle_left_type = 'AUTO'
                        keyframe.handle_right_type = 'AUTO'
                    else:
                        # Use LINEAR interpolation for uniform time steps
                        keyframe.interpolation = 'LINEAR'
        
        success_msg = f"Successfully animated object with {len(positions)} keyframes"
        if times is not None:
            success_msg += f" using variable time steps (range: {times[0]:.3f}s - {times[-1]:.3f}s)"
        else:
            success_msg += " using uniform time steps"
        print(success_msg)
        
        return True
    
    def animate_trajectory_growth(self, trajectory_obj, frames: int = 100, times: Optional[np.ndarray] = None):
        """Animate trajectory appearing over time using build modifier.
        
        Args:
            trajectory_obj: The trajectory object to animate
            frames: Total number of animation frames
            times: Optional array of simulation times for variable step timing
        """
        try:
            # Add build modifier to make trajectory appear over time
            build_modifier = trajectory_obj.modifiers.new(name="Build", type='BUILD')
            build_modifier.frame_start = 1
            build_modifier.frame_duration = frames
            build_modifier.use_random_order = False
            
            # If variable timing is provided, we could potentially animate the build
            # modifier's frame_duration based on the actual time distribution
            # For now, we keep the uniform growth but this could be enhanced
            if times is not None:
                print(f"Trajectory growth using variable timing over {frames} frames")
            
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
    
    def analyze_timing_distribution(self, times: Optional[np.ndarray], total_frames: int = 250) -> dict:
        """Analyze the timing distribution for debugging variable step sizes.
        
        Args:
            times: Array of simulation times
            total_frames: Total animation frames
            
        Returns:
            Dictionary with timing analysis
        """
        if times is None:
            return {"type": "uniform", "message": "No time data provided - using uniform spacing"}
        
        time_duration = times[-1] - times[0]
        if time_duration <= 0:
            return {"type": "error", "message": "Invalid time duration"}
        
        # Calculate time steps
        dt_values = np.diff(times)
        dt_mean = np.mean(dt_values)
        dt_std = np.std(dt_values)
        dt_min = np.min(dt_values)
        dt_max = np.max(dt_values)
        
        # Calculate frame distribution
        frame_times = ((times - times[0]) / time_duration) * (total_frames - 1) + 1
        frame_steps = np.diff(frame_times)
        
        analysis = {
            "type": "variable",
            "total_points": len(times),
            "time_duration": time_duration,
            "dt_stats": {
                "mean": dt_mean,
                "std": dt_std,
                "min": dt_min,
                "max": dt_max,
                "coefficient_of_variation": dt_std / dt_mean if dt_mean > 0 else 0
            },
            "frame_distribution": {
                "min_frame_step": np.min(frame_steps),
                "max_frame_step": np.max(frame_steps),
                "mean_frame_step": np.mean(frame_steps)
            }
        }
        
        # Determine if timing is effectively uniform
        cv_threshold = 0.1  # 10% coefficient of variation threshold
        if analysis["dt_stats"]["coefficient_of_variation"] < cv_threshold:
            analysis["message"] = f"Timing is nearly uniform (CV: {analysis['dt_stats']['coefficient_of_variation']:.3f})"
        else:
            analysis["message"] = f"Variable timing detected (CV: {analysis['dt_stats']['coefficient_of_variation']:.3f})"
        
        return analysis
    
    def animate_camera_tracking_object(self, target_object: Any, 
                                     target_positions: np.ndarray,
                                     total_frames: int = 250,
                                     frame_start: int = 1,
                                     times: Optional[np.ndarray] = None,
                                     camera_offset: np.ndarray = None,
                                     look_ahead_frames: int = 0,
                                     smooth_tracking: bool = True) -> bool:
        """Animate camera to track a moving object.
        
        Args:
            target_object: Blender object to track
            target_positions: (N, 3) array of target object positions
            total_frames: Total number of animation frames
            frame_start: Starting frame number
            times: Optional (N,) array of simulation times for variable step timing
            camera_offset: (3,) array of camera offset from target (default: behind and above)
            look_ahead_frames: Number of frames to look ahead for smoother tracking
            smooth_tracking: Whether to use smooth tracking or direct following
            
        Returns:
            True if animation was successful, False otherwise
        """
        
        if target_positions is None or len(target_positions) == 0:
            print("Error: No target positions provided for camera tracking")
            return False
        
        try:
            # Get or create camera
            if 'Camera' in bpy.data.objects:
                camera = bpy.data.objects['Camera']
            else:
                # Create camera if it doesn't exist
                bpy.ops.object.camera_add()
                camera = bpy.context.active_object
                camera.name = 'TrackingCamera'
            
            # Set default camera offset if not provided
            if camera_offset is None:
                # Position camera behind and above the spacecraft
                camera_offset = np.array([0, -50, 20]) * self.scale_factor
            
            # Scale positions
            scaled_positions = target_positions * self.scale_factor
            
            # Clear existing animation data and constraints
            camera.animation_data_clear()
            camera.constraints.clear()
            
            # Set up time mapping for variable step sizes
            if times is not None:
                time_duration = times[-1] - times[0]
                if time_duration > 0:
                    # Map simulation time to frame numbers
                    frame_times = ((times - times[0]) / time_duration) * (total_frames - 1) + frame_start
                    print(f"Using variable time steps for camera animation: {len(times)} time points over {total_frames} frames")
                else:
                    frame_times = np.linspace(frame_start, frame_start + total_frames - 1, len(scaled_positions))
            else:
                frame_times = np.linspace(frame_start, frame_start + total_frames - 1, len(scaled_positions))
            
            # Add Track To constraint to make camera always look at target
            track_constraint = camera.constraints.new(type='TRACK_TO')
            track_constraint.target = target_object
            track_constraint.track_axis = 'TRACK_NEGATIVE_Z'  # Camera looks down -Z axis
            track_constraint.up_axis = 'UP_Y'  # Y is up
            track_constraint.name = "TrackToSpacecraft"
            
            # Animate camera position
            for i, (pos, frame_time) in enumerate(zip(scaled_positions, frame_times)):
                # Calculate look-ahead position for smoother tracking
                if look_ahead_frames > 0 and i + look_ahead_frames < len(scaled_positions):
                    target_pos = scaled_positions[i + look_ahead_frames]
                else:
                    target_pos = pos
                
                # Position camera with offset from target
                camera_pos = target_pos + camera_offset
                camera.location = camera_pos
                camera.keyframe_insert(data_path="location", frame=int(frame_time))
            
            # Set interpolation mode for smooth camera movement
            # Use BEZIER interpolation for variable time steps, LINEAR for uniform
            if camera.animation_data and camera.animation_data.action:
                for fcurve in camera.animation_data.action.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        if times is not None:
                            # Use BEZIER interpolation for variable time steps for smoother camera movement
                            keyframe.interpolation = 'BEZIER'
                            keyframe.handle_left_type = 'AUTO'
                            keyframe.handle_right_type = 'AUTO'
                        else:
                            # Use LINEAR interpolation for uniform time steps
                            keyframe.interpolation = 'LINEAR'
            
            success_msg = f"Camera tracking setup complete with {len(scaled_positions)} keyframes"
            if times is not None:
                success_msg += f" using variable timing (range: {times[0]:.3f}s - {times[-1]:.3f}s)"
            else:
                success_msg += " using uniform timing"
            print(success_msg)
            return True
            
        except Exception as e:
            print(f"Error setting up camera tracking: {e}")
            return False
