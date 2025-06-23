"""
Blender scene setup and management utilities.
"""

import sys
import os
from typing import Optional, Dict, Any, Tuple

try:
    import bpy
    import bmesh
    from mathutils import Vector, Euler

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


class BlenderScene:
    """Manages Blender scene setup and configuration for trajectory rendering."""

    def __init__(self, scene_name: str = "BioInspired_Scene"):
        """Initialize Blender scene manager."""
        if not BLENDER_AVAILABLE:
            raise ImportError("Blender Python API not available")

        self.scene_name = scene_name
        self.scene = None
        self.camera = None
        self.lights = []

    def setup_scene(self, clear_existing: bool = True) -> bpy.types.Scene:
        """Set up a new Blender scene for rendering."""
        # if clear_existing:?
        self.clear_scene()

        # Rename the current scene
        if bpy.context.scene.name != self.scene_name:
            bpy.context.scene.name = self.scene_name
        self.scene = bpy.context.scene

        # Setup camera
        print("Setting up camera...")
        self.setup_camera()

        # Setup lighting
        self.setup_lighting()

        # Configure render settings
        self.setup_render_settings()

        return self.scene

    def clear_scene(self):
        """Clear all objects from the scene."""
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

    def setup_camera(
        self,
        location: Tuple[float, float, float] = (10, -10, 8),
        rotation: Tuple[float, float, float] = (1.1, 0, 0.8),
    ):
        """Set up camera for optimal trajectory viewing."""
        # Create camera
        bpy.ops.object.camera_add(
            enter_editmode=False,
            align="VIEW",
            location=location,
            rotation=rotation,
            scale=(1, 1, 1),
        )
        self.camera = bpy.context.object
        self.camera.name = "TrajectoryCamera"

        # Set camera properties
        self.camera.data.lens = 50
        self.camera.data.clip_end = 1000

        # Set as active camera
        self.scene.camera = self.camera

        return self.camera

    def setup_lighting(self):
        """Set up lighting for molecular visualization."""
        # Key light
        bpy.ops.object.light_add(type="SUN", location=(5, 5, 10))
        key_light = bpy.context.object
        key_light.name = "KeyLight"
        key_light.data.energy = 3
        self.lights.append(key_light)

        # Fill light
        bpy.ops.object.light_add(type="AREA", location=(-5, -5, 5))
        fill_light = bpy.context.object
        fill_light.name = "FillLight"
        fill_light.data.energy = 1
        fill_light.data.size = 5
        self.lights.append(fill_light)

        # Rim light
        bpy.ops.object.light_add(type="SPOT", location=(0, -10, 2))
        rim_light = bpy.context.object
        rim_light.name = "RimLight"
        rim_light.data.energy = 2
        rim_light.data.spot_size = 1.2
        self.lights.append(rim_light)

        return self.lights

    def setup_render_settings(
        self, resolution: Tuple[int, int] = (1920, 1080), samples: int = 128
    ):
        """Configure render settings for high-quality output."""
        scene = self.scene

        # Resolution
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        scene.render.resolution_percentage = 100

        # Render engine
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True

        # Output settings
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = True

    def add_trajectory_path(
        self,
        trajectory_data,
        name: str = "trajectory",
        material_name: str = "trajectory_material",
    ):
        """Add a trajectory path to the scene."""
        # Create curve from trajectory data
        curve_data = bpy.data.curves.new(name, type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.resolution_u = 2

        # Create spline
        polyline = curve_data.splines.new("POLY")
        polyline.points.add(len(trajectory_data) - 1)

        # Add points
        for i, point in enumerate(trajectory_data):
            polyline.points[i].co = (point[0], point[1], point[2], 1)

        # Create object
        curve_obj = bpy.data.objects.new(name, curve_data)
        self.scene.collection.objects.link(curve_obj)

        # Set curve properties
        curve_data.bevel_depth = 0.1
        curve_data.bevel_resolution = 4

        return curve_obj

    def frame_all_objects(self):
        """Frame all objects in the camera view."""
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.view3d.camera_to_view_selected()

    def render_image(self, output_path: str):
        """Render the current scene to an image."""
        self.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        return output_path
