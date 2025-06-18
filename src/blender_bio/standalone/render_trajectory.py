"""
Standalone script for rendering trajectories in Blender.

Usage:
    # From command line
    blender --background --python render_trajectory.py -- --trajectory_id 1 --output_path ./output/

    # From within Blender
    exec(open("render_trajectory.py").read())
"""

import sys
import os
import argparse
from pathlib import Path

# Add project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("Warning: Not running in Blender environment")

# Import our modules
from blender_bio.core.scene_setup import BlenderScene
from blender_bio.trajectory.path_renderer import TrajectoryRenderer


def render_trajectory_from_database(trajectory_id: int, output_path: str, 
                                  render_type: str = "path"):
    """Render a trajectory from the database."""
    if not BLENDER_AVAILABLE:
        raise RuntimeError("This script must be run within Blender")
    
    print(f"Rendering trajectory {trajectory_id}...")
    
    # Setup scene
    scene_manager = BlenderScene("TrajectoryRender")
    scene_manager.setup_scene()
    
    # Setup renderer
    renderer = TrajectoryRenderer(scene_manager)
    
    # Load trajectory from database
    try:
        trajectory_data, trajectory_obj = renderer.load_trajectory_from_database(trajectory_id)
        print(f"Loaded trajectory with {len(trajectory_data)} points")
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return False
    
    # Render based on type
    if render_type == "path":
        renderer.render_trajectory_path(trajectory_data, f"trajectory_{trajectory_id}")
    elif render_type == "spheres":
        renderer.render_trajectory_spheres(trajectory_data, f"trajectory_{trajectory_id}")
    elif render_type == "both":
        renderer.render_trajectory_path(trajectory_data, f"trajectory_{trajectory_id}_path")
        renderer.render_trajectory_spheres(trajectory_data, f"trajectory_{trajectory_id}_spheres", 
                                         subsample=10)
    
    # Add start/end markers
    renderer.render_start_end_markers(trajectory_data, f"trajectory_{trajectory_id}_markers")
    
    # Frame all objects
    scene_manager.frame_all_objects()
    
    # Render image
    output_file = os.path.join(output_path, f"trajectory_{trajectory_id}.png")
    os.makedirs(output_path, exist_ok=True)
    scene_manager.render_image(output_file)
    
    print(f"Rendered trajectory to: {output_file}")
    return True


def render_multiple_trajectories(trajectory_ids: list, output_path: str):
    """Render multiple trajectories in the same scene."""
    if not BLENDER_AVAILABLE:
        raise RuntimeError("This script must be run within Blender")
    
    print(f"Rendering {len(trajectory_ids)} trajectories...")
    
    # Setup scene
    scene_manager = BlenderScene("MultiTrajectoryRender")
    scene_manager.setup_scene()
    
    # Setup renderer
    renderer = TrajectoryRenderer(scene_manager)
    
    # Load and render all trajectories
    trajectories = []
    names = []
    
    for trajectory_id in trajectory_ids:
        try:
            trajectory_data, trajectory_obj = renderer.load_trajectory_from_database(trajectory_id)
            trajectories.append(trajectory_data)
            names.append(f"trajectory_{trajectory_id}")
            print(f"Loaded trajectory {trajectory_id} with {len(trajectory_data)} points")
        except Exception as e:
            print(f"Error loading trajectory {trajectory_id}: {e}")
            continue
    
    if not trajectories:
        print("No trajectories could be loaded")
        return False
    
    # Render all trajectories
    renderer.render_multiple_trajectories(trajectories, names)
    
    # Frame all objects
    scene_manager.frame_all_objects()
    
    # Render image
    output_file = os.path.join(output_path, f"multi_trajectory_{'_'.join(map(str, trajectory_ids))}.png")
    os.makedirs(output_path, exist_ok=True)
    scene_manager.render_image(output_file)
    
    print(f"Rendered multiple trajectories to: {output_file}")
    return True


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Render BioInspired trajectories in Blender")
    parser.add_argument("--trajectory_id", type=int, help="Single trajectory ID to render")
    parser.add_argument("--trajectory_ids", nargs="+", type=int, help="Multiple trajectory IDs to render")
    parser.add_argument("--output_path", default="./renders/", help="Output directory for rendered images")
    parser.add_argument("--render_type", choices=["path", "spheres", "both"], default="path",
                       help="Type of rendering")
    
    # Parse arguments (handle Blender's -- separator)
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]
    
    args = parser.parse_args(argv)
    
    # Validate arguments
    if not args.trajectory_id and not args.trajectory_ids:
        print("Error: Must specify either --trajectory_id or --trajectory_ids")
        return False
    
    # Render trajectories
    try:
        if args.trajectory_id:
            return render_trajectory_from_database(args.trajectory_id, args.output_path, args.render_type)
        elif args.trajectory_ids:
            return render_multiple_trajectories(args.trajectory_ids, args.output_path)
    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # If running in Blender, we can also run interactively
    if BLENDER_AVAILABLE and len(sys.argv) == 1:
        print("Running in interactive mode...")
        print("Example usage:")
        print("  render_trajectory_from_database(1, './output/')")
        print("  render_multiple_trajectories([1, 2, 3], './output/')")
    else:
        success = main()
        sys.exit(0 if success else 1)
