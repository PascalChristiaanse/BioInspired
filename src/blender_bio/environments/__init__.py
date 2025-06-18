"""
Space environment rendering module for creating realistic space scenes.
"""

import bpy
import bmesh
import mathutils
import random
from typing import Dict, List, Tuple, Optional

class SpaceEnvironmentRenderer:
    """Renders space environments including Earth, stars, and lighting."""
    
    def __init__(self):
        self.environment_objects = {}
        
    def setup_space_scene(self, include_earth: bool = True, 
                         include_stars: bool = True,
                         include_sun: bool = True):
        """Set up a complete space environment."""
        # Clear existing scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Set up world material for space
        self._setup_space_world()
        
        if include_earth:
            self.create_earth()
            
        if include_stars:
            self.create_star_field()
            
        if include_sun:
            self.create_sun_lighting()
            
        # Set camera for space view
        self._setup_space_camera()
    
    def create_earth(self, position: Tuple[float, float, float] = (0, -20, 0),
                    radius: float = 6.0) -> bpy.types.Object:
        """Create Earth in the background."""
        bpy.ops.mesh.primitive_uv_sphere_add(location=position, radius=radius)
        earth = bpy.context.active_object
        earth.name = "Earth"
        
        # Apply Earth material
        self._apply_earth_material(earth)
        
        self.environment_objects["Earth"] = earth
        return earth
    
    def create_star_field(self, count: int = 1000, 
                         distance: float = 100.0) -> List[bpy.types.Object]:
        """Create a field of background stars."""
        stars = []
        
        for i in range(count):
            # Random position on sphere
            phi = random.uniform(0, 2 * 3.14159)
            theta = random.uniform(0, 3.14159)
            
            x = distance * math.sin(theta) * math.cos(phi)
            y = distance * math.sin(theta) * math.sin(phi)
            z = distance * math.cos(theta)
            
            # Create small sphere for star
            bpy.ops.mesh.primitive_uv_sphere_add(
                location=(x, y, z), 
                radius=random.uniform(0.01, 0.05)
            )
            star = bpy.context.active_object
            star.name = f"Star_{i:04d}"
            
            # Apply star material
            self._apply_star_material(star, brightness=random.uniform(0.5, 2.0))
            
            stars.append(star)
        
        # Group stars
        bpy.ops.object.select_all(action='DESELECT')
        for star in stars:
            star.select_set(True)
        
        bpy.ops.object.join()
        star_field = bpy.context.active_object
        star_field.name = "StarField"
        
        self.environment_objects["StarField"] = star_field
        return [star_field]
    
    def create_sun_lighting(self, position: Tuple[float, float, float] = (50, 50, 30),
                          strength: float = 5.0) -> bpy.types.Object:
        """Create sun lighting."""
        bpy.ops.object.light_add(type='SUN', location=position)
        sun = bpy.context.active_object
        sun.name = "Sun"
        
        # Configure sun properties
        sun.data.energy = strength
        sun.data.color = (1.0, 0.95, 0.8)  # Slightly warm white
        sun.data.angle = 0.01  # Sharp shadows
        
        # Point towards origin
        direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector(position)
        sun.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        
        self.environment_objects["Sun"] = sun
        return sun
    
    def add_ambient_lighting(self, strength: float = 0.1):
        """Add subtle ambient lighting for space."""
        world = bpy.data.worlds.get("World")
        if world and world.use_nodes:
            # Add subtle blue ambient light
            background = world.node_tree.nodes.get("Background")
            if background:
                background.inputs[0].default_value = (0.01, 0.02, 0.04, 1.0)  # Very dark blue
                background.inputs[1].default_value = strength
    
    def _setup_space_world(self):
        """Set up world material for space."""
        world = bpy.data.worlds.get("World")
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        world.node_tree.nodes.clear()
        
        # Background shader
        background = world.node_tree.nodes.new(type='ShaderNodeBackground')
        background.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)  # Pure black
        background.inputs[1].default_value = 0.0  # No emission
        
        # World output
        output = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
        world.node_tree.links.new(background.outputs[0], output.inputs[0])
    
    def _setup_space_camera(self):
        """Set up camera for space scene."""
        # Delete existing camera
        if bpy.data.objects.get("Camera"):
            bpy.data.objects.remove(bpy.data.objects["Camera"], do_unlink=True)
        
        # Create new camera
        bpy.ops.object.camera_add(location=(0, -10, 5))
        camera = bpy.context.active_object
        camera.name = "SpaceCamera"
        
        # Point camera at origin
        constraint = camera.constraints.new(type='TRACK_TO')
        
        # Set as active camera
        bpy.context.scene.camera = camera
    
    def _apply_earth_material(self, obj: bpy.types.Object):
        """Apply Earth material with continents and oceans."""
        mat = bpy.data.materials.new(name="EarthMaterial")
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        
        # Create principled BSDF
        principled = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # Add noise texture for continents
        noise = mat.node_tree.nodes.new(type='ShaderNodeTexNoise')
        noise.inputs[2].default_value = 5.0  # Scale
        noise.inputs[3].default_value = 0.5  # Detail
        
        # Color ramp for land/ocean
        color_ramp = mat.node_tree.nodes.new(type='ShaderNodeValToRGB')
        color_ramp.color_ramp.elements[0].color = (0.1, 0.3, 0.8, 1.0)  # Ocean blue
        color_ramp.color_ramp.elements[1].color = (0.2, 0.6, 0.1, 1.0)  # Land green
        
        # Connect nodes
        mat.node_tree.links.new(noise.outputs[0], color_ramp.inputs[0])
        mat.node_tree.links.new(color_ramp.outputs[0], principled.inputs[0])
        
        # Add some emission for atmosphere glow
        principled.inputs[17].default_value = (0.5, 0.7, 1.0)  # Blue glow
        principled.inputs[18].default_value = 0.1  # Subtle emission
        
        # World output
        output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(principled.outputs[0], output.inputs[0])
        
        obj.data.materials.append(mat)
    
    def _apply_star_material(self, obj: bpy.types.Object, brightness: float = 1.0):
        """Apply star material with emission."""
        mat = bpy.data.materials.new(name=f"StarMaterial_{obj.name}")
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        
        # Emission shader for bright stars
        emission = mat.node_tree.nodes.new(type='ShaderNodeEmission')
        emission.inputs[0].default_value = (1.0, 1.0, 0.9, 1.0)  # Slightly warm white
        emission.inputs[1].default_value = brightness
        
        # Output
        output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(emission.outputs[0], output.inputs[0])
        
        obj.data.materials.append(mat)


import math  # Import needed for star field creation
