"""
Standalone script for rendering trajectories in Blender.

Usage from within Blender
    exec(open("render_trajectory.py").read())
"""

import sys
import os
from pathlib import Path

# Add project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import bpy
except ImportError:
    raise ImportError(
        "This script must be run within Blender. Please run it using Blender's Python environment."
    )


# Import our modules
from src.blender_bio.core.scene_setup import BlenderScene
from src.blender_bio.trajectory.path_renderer import TrajectoryRenderer


def render_trajectory_from_database(
    trajectory_id: int, output_path: str, render_type: str = "path"
):
    """Render a trajectory from the database."""
    print(f"Rendering trajectory {trajectory_id}...")

    # Setup scene
    scene_manager = BlenderScene("TrajectoryRender")
    scene_manager.setup_scene()
    # Setup renderer
    renderer = TrajectoryRenderer(scene_manager)

    # Load trajectory from database
    try:
        trajectory_data, trajectory_obj = renderer.load_trajectory_from_database(
            trajectory_id
        )
        print(f"Loaded trajectory with {len(trajectory_data)} points")
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return False

    # Render based on type
    if render_type == "path":
        renderer.render_trajectory_path(trajectory_data, f"trajectory_{trajectory_id}")
    elif render_type == "spheres":
        renderer.render_trajectory_spheres(
            trajectory_data, f"trajectory_{trajectory_id}"
        )
    elif render_type == "both":
        renderer.render_trajectory_path(
            trajectory_data, f"trajectory_{trajectory_id}_path"
        )
        renderer.render_trajectory_spheres(
            trajectory_data, f"trajectory_{trajectory_id}_spheres", subsample=10
        )

    # Add start/end markers
    renderer.render_start_end_markers(
        trajectory_data, f"trajectory_{trajectory_id}_markers"
    )

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
            trajectory_data, trajectory_obj = renderer.load_trajectory_from_database(
                trajectory_id
            )
            trajectories.append(trajectory_data)
            names.append(f"trajectory_{trajectory_id}")
            print(
                f"Loaded trajectory {trajectory_id} with {len(trajectory_data)} points"
            )
        except Exception as e:
            print(f"Error loading trajectory {trajectory_id}: {e}")
            continue

    if not trajectories:
        print(" Ptrajectories could be loaded")
        return False

    # Render all trajectories
    renderer.render_multiple_trajectories(trajectories, names)

    # Frame all objects
    scene_manager.frame_all_objects()

    # Render image
    output_file = os.path.join(
        output_path, f"multi_trajectory_{'_'.join(map(str, trajectory_ids))}.png"
    )
    os.makedirs(output_path, exist_ok=True)
    scene_manager.render_image(output_file)

    print(f"Rendered multiple trajectories to: {output_file}")
    return True


def main():
    """Main function for interactive usage with input()."""
    print("=== Blender Trajectory Renderer ===")
    ids_str = input("Enter trajectory ID(s) (comma-separated for multiple): ").strip()
    # Split by comma, strip spaces, and filter valid integers
    id_list = [int(x) for x in ids_str.split(",") if x.strip().isdigit()]
    output_path = input("Enter output path [./renders/]: ").strip() or "./renders/"
    render_type = "path"
    if len(id_list) == 1:
        render_type = (
            input("Render type (fpath/spheres/both) [path]: ").strip() or "path"
        )
        success = render_trajectory_from_database(id_list[0], output_path, render_type)
    elif len(id_list) > 1:
        success = render_multiple_trajectories(id_list, output_path)
    else:
        print("No valid trajectory IDs entered.")
        return False
    return success


if __name__ == "<run_path>":
    # If running in Blender, we can also run interactively
    print("Render trajectory script executed in Blender environment.")
    success = main()
    print("Rendering completed." if success else "Rendering failed.")
    # sys.exit(0 if success else 1) 
