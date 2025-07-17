"""
Standalone script for animating trajectories in Blender.

Usage from within Blender:
    exec(open("animate_trajectory.py").read())
"""

import sys
from pathlib import Path

# Add project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import bpy
    import numpy as np
    BLENDER_AVAILABLE = True
except ImportError:
    raise ImportError(
        "This script must be run within Blender. Please run it using Blender's Python environment."
    )

# Import our modules after Blender is confirmed available
from src.blender_bio.core.scene_setup import BlenderScene
from src.blender_bio.trajectory.animation import AnimationManager
from src.blender_bio.trajectory.path_renderer import TrajectoryRenderer


def animate_trajectory_from_database(
    trajectory_id: int, 
    animation_frames: int = 250,
    show_trajectory_path: bool = True,
    show_spacecraft: bool = True,
    show_camera_animation: bool = False
):
    """Animate a trajectory from the database."""
    print(f"Animating trajectory {trajectory_id}...")

    # Setup scene
    scene_manager = BlenderScene("TrajectoryAnimation")
    scene_manager.setup_scene()
    
    # Setup animation manager
    animator = AnimationManager()
    animator.set_animation_range(1, animation_frames)
    
    # Setup trajectory renderer for path visualization
    if show_trajectory_path:
        renderer = TrajectoryRenderer(scene_manager)
        
        # Load trajectory from database
        try:
            trajectory_data, trajectory_record = renderer.load_trajectory_from_database(trajectory_id)
            print(f"Loaded trajectory with {len(trajectory_data)} points")
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            return False
    else:
        # Load trajectory data directly through animator
        from src.bioinspired.data.services import get_trajectory
        trajectory_record = get_trajectory(trajectory_id)
        if not trajectory_record:
            print(f"Error: Trajectory with ID {trajectory_id} not found.")
            return False
        
        trajectory_data, velocities, times = animator.extract_trajectory_data(trajectory_record)
        if trajectory_data is None:
            print(f"Error: Could not extract trajectory data for ID {trajectory_id}")
            return False
    
    # Create and show trajectory path
    if show_trajectory_path:
        try:
            trajectory_path = renderer.render_trajectory_path(
                trajectory_data, 
                f"trajectory_{trajectory_id}_path",
                path_width=0.02,
                color=(0.3, 0.7, 0.9, 1.0)  # Light blue
            )
            
            # Animate trajectory growth
            animator.animate_trajectory_growth(trajectory_path, animation_frames)
            print("Created animated trajectory path")
        except Exception as e:
            print(f"Error creating trajectory path: {e}")
    
    # Create and animate spacecraft
    if show_spacecraft:
        try:
            # Extract full trajectory data for spacecraft animation
            positions, velocities, times = animator.extract_trajectory_data(trajectory_record)
            
            if positions is not None:
                # Create spacecraft object
                spacecraft = animator.create_spacecraft_object(
                    f"spacecraft_{trajectory_id}", 
                    size=0.5 * animator.scale_factor * 1000  # Adjust size based on scale
                )
                
                # Animate spacecraft along trajectory
                success = animator.animate_object_along_trajectory(
                    spacecraft, 
                    positions, 
                    velocities, 
                    times, 
                    animation_frames
                )
                
                if success:
                    print("Created animated spacecraft")
                else:
                    print("Failed to animate spacecraft")
            else:
                print("No position data available for spacecraft animation")
        except Exception as e:
            print(f"Error creating spacecraft animation: {e}")
    
    # Animate camera following trajectory
    if show_camera_animation and trajectory_data is not None:
        try:
            success = animator.animate_camera_path(
                trajectory_data, 
                animation_frames,
                camera_offset=np.array([0, -20, 10]),  # Camera behind and above
                target_offset=np.array([0, 10, 0])     # Look slightly ahead
            )
            
            if success:
                print("Created camera animation")
            else:
                print("Failed to animate camera")
        except Exception as e:
            print(f"Error creating camera animation: {e}")
    
    # Add start/end markers
    if show_trajectory_path:
        try:
            renderer.render_start_end_markers(
                trajectory_data, 
                f"trajectory_{trajectory_id}_markers",
                start_color=(0.0, 1.0, 0.0, 1.0),  # Green
                end_color=(1.0, 0.0, 0.0, 1.0),    # Red
                marker_size=0.3 * animator.scale_factor * 1000
            )
            print("Added start/end markers")
        except Exception as e:
            print(f"Error adding markers: {e}")
    
    # Frame all objects
    scene_manager.frame_all_objects()
    
    print(f"Animation setup complete for trajectory {trajectory_id}")
    print(f"Animation frames: {animation_frames}")
    print("Use the spacebar to play the animation in Blender")
    
    return True


def animate_multiple_trajectories(
    trajectory_ids: list, 
    animation_frames: int = 250,
    show_trajectory_paths: bool = True,
    show_spacecraft: bool = True
):
    """Animate multiple trajectories in the same scene."""
    print(f"Animating {len(trajectory_ids)} trajectories...")

    # Setup scene
    scene_manager = BlenderScene("MultiTrajectoryAnimation")
    scene_manager.setup_scene()
    
    # Setup animation manager
    animator = AnimationManager()
    animator.set_animation_range(1, animation_frames)
    
    # Setup trajectory renderer
    renderer = TrajectoryRenderer(scene_manager)
    
    # Colors for different trajectories
    colors = [
        (0.8, 0.2, 0.2, 1.0),  # Red
        (0.2, 0.8, 0.2, 1.0),  # Green
        (0.2, 0.2, 0.8, 1.0),  # Blue
        (0.8, 0.8, 0.2, 1.0),  # Yellow
        (0.8, 0.2, 0.8, 1.0),  # Magenta
        (0.2, 0.8, 0.8, 1.0),  # Cyan
    ]
    
    animated_objects = []
    
    for i, trajectory_id in enumerate(trajectory_ids):
        color = colors[i % len(colors)]
        
        try:
            # Load trajectory data
            trajectory_data, trajectory_record = renderer.load_trajectory_from_database(trajectory_id)
            if trajectory_data is None:
                print(f"Skipping trajectory {trajectory_id} - no data")
                continue
            
            print(f"Loaded trajectory {trajectory_id} with {len(trajectory_data)} points")
            
            # Create trajectory path
            if show_trajectory_paths:
                trajectory_path = renderer.render_trajectory_path(
                    trajectory_data, 
                    f"trajectory_{trajectory_id}_path",
                    path_width=0.02,
                    color=color
                )
                
                # Animate trajectory growth with offset
                animator.animate_trajectory_growth(trajectory_path, animation_frames)
                animated_objects.append(trajectory_path)
            
            # Create and animate spacecraft
            if show_spacecraft:
                positions, velocities, times = animator.extract_trajectory_data(trajectory_record)
                
                if positions is not None:
                    spacecraft = animator.create_spacecraft_object(
                        f"spacecraft_{trajectory_id}", 
                        size=0.3 * animator.scale_factor * 1000
                    )
                    
                    # Set spacecraft color
                    spacecraft.data.materials[0].node_tree.nodes["Principled BSDF"].inputs[0].default_value = color
                    
                    # Animate spacecraft
                    animator.animate_object_along_trajectory(
                        spacecraft, 
                        positions, 
                        velocities, 
                        times, 
                        animation_frames
                    )
                    animated_objects.append(spacecraft)
            
            # Add markers
            renderer.render_start_end_markers(
                trajectory_data, 
                f"trajectory_{trajectory_id}_markers",
                start_color=(0.0, 1.0, 0.0, 1.0),
                end_color=(1.0, 0.0, 0.0, 1.0),
                marker_size=0.2 * animator.scale_factor * 1000
            )
            
        except Exception as e:
            print(f"Error processing trajectory {trajectory_id}: {e}")
            continue
    
    if not animated_objects:
        print("No trajectories could be animated")
        return False
    
    # Frame all objects
    scene_manager.frame_all_objects()
    
    print(f"Animation setup complete for {len(animated_objects)} objects")
    print(f"Animation frames: {animation_frames}")
    print("Use the spacebar to play the animation in Blender")
    
    return True


def create_comparison_animation(
    trajectory_ids: list,
    animation_frames: int = 250,
    sync_timing: bool = True
):
    """Create a comparison animation of multiple trajectories with synchronized timing."""
    print(f"Creating comparison animation for {len(trajectory_ids)} trajectories...")
    
    # Setup scene
    scene_manager = BlenderScene("TrajectoryComparison")
    scene_manager.setup_scene()
    
    # Setup animation manager
    animator = AnimationManager()
    animator.set_animation_range(1, animation_frames)
    
    # Load all trajectory data first
    trajectory_data_list = []
    from src.bioinspired.data.services import get_trajectory
    
    for trajectory_id in trajectory_ids:
        try:
            trajectory_record = get_trajectory(trajectory_id)
            if trajectory_record:
                positions, velocities, times = animator.extract_trajectory_data(trajectory_record)
                if positions is not None:
                    trajectory_data_list.append({
                        'id': trajectory_id,
                        'positions': positions,
                        'velocities': velocities,
                        'times': times,
                        'record': trajectory_record
                    })
        except Exception as e:
            print(f"Error loading trajectory {trajectory_id}: {e}")
    
    if not trajectory_data_list:
        print("No valid trajectories found")
        return False
    
    # Create spacecraft for each trajectory
    colors = [
        (0.8, 0.2, 0.2, 1.0),  # Red
        (0.2, 0.8, 0.2, 1.0),  # Green
        (0.2, 0.2, 0.8, 1.0),  # Blue
        (0.8, 0.8, 0.2, 1.0),  # Yellow
    ]
    
    for i, traj_data in enumerate(trajectory_data_list):
        color = colors[i % len(colors)]
        
        # Create spacecraft
        spacecraft = animator.create_spacecraft_object(
            f"spacecraft_{traj_data['id']}", 
            size=0.4 * animator.scale_factor * 1000
        )
        
        # Set color
        spacecraft.data.materials[0].node_tree.nodes["Principled BSDF"].inputs[0].default_value = color
        
        # Animate spacecraft
        animator.animate_object_along_trajectory(
            spacecraft,
            traj_data['positions'],
            traj_data['velocities'],
            traj_data['times'] if not sync_timing else None,  # Use None for synchronized timing
            animation_frames
        )
    
    # Frame all objects
    scene_manager.frame_all_objects()
    
    print(f"Comparison animation created with {len(trajectory_data_list)} spacecraft")
    print("Use the spacebar to play the animation in Blender")
    
    return True


def main():
    """Main function for interactive usage."""
    print("=== Blender Trajectory Animator ===")
    print("1. Animate single trajectory")
    print("2. Animate multiple trajectories")
    print("3. Create comparison animation")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        trajectory_id = int(input("Enter trajectory ID: ").strip())
        frames = int(input("Enter animation frames [250]: ").strip() or "250")
        
        show_path = input("Show trajectory path? (y/n) [y]: ").strip().lower() in ('y', 'yes', '')
        show_spacecraft = input("Show animated spacecraft? (y/n) [y]: ").strip().lower() in ('y', 'yes', '')
        show_camera = input("Animate camera? (y/n) [n]: ").strip().lower() in ('y', 'yes')
        
        success = animate_trajectory_from_database(
            trajectory_id, 
            frames, 
            show_path, 
            show_spacecraft, 
            show_camera
        )
        
    elif choice == "2":
        ids_str = input("Enter trajectory IDs (comma-separated): ").strip()
        trajectory_ids = [int(x.strip()) for x in ids_str.split(",") if x.strip().isdigit()]
        frames = int(input("Enter animation frames [250]: ").strip() or "250")
        
        show_paths = input("Show trajectory paths? (y/n) [y]: ").strip().lower() in ('y', 'yes', '')
        show_spacecraft = input("Show animated spacecraft? (y/n) [y]: ").strip().lower() in ('y', 'yes', '')
        
        success = animate_multiple_trajectories(
            trajectory_ids, 
            frames, 
            show_paths, 
            show_spacecraft
        )
        
    elif choice == "3":
        ids_str = input("Enter trajectory IDs (comma-separated): ").strip()
        trajectory_ids = [int(x.strip()) for x in ids_str.split(",") if x.strip().isdigit()]
        frames = int(input("Enter animation frames [250]: ").strip() or "250")
        
        sync_timing = input("Synchronize timing? (y/n) [y]: ").strip().lower() in ('y', 'yes', '')
        
        success = create_comparison_animation(
            trajectory_ids, 
            frames, 
            sync_timing
        )
        
    else:
        print("Invalid choice")
        return False
    
    return success


if __name__ == "<run_path>":
    print("Trajectory animation script executed in Blender environment.")
    success = main()
    print("Animation setup completed." if success else "Animation setup failed.")
