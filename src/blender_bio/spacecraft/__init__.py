"""
Spacecraft rendering module for 3D visualization of spacecraft and docking ports.
"""

import bpy
import bmesh
import mathutils
import numpy as np
from typing import Dict, List, Tuple, Optional

class SpacecraftRenderer:
    """Renders spacecraft objects in Blender scenes."""
    
    def __init__(self, scene_name: str = "SpacecraftScene"):
        self.scene_name = scene_name
        self.spacecraft_objects = {}
        
    def create_basic_spacecraft(self, name: str = "Spacecraft", 
                              position: Tuple[float, float, float] = (0, 0, 0),
                              scale: float = 1.0) -> bpy.types.Object:
        """Create a basic spacecraft model."""
        # Clear existing mesh
        bpy.ops.mesh.primitive_cube_add(location=position)
        spacecraft = bpy.context.active_object
        spacecraft.name = name
        
        # Enter edit mode to modify the mesh
        bpy.context.view_layer.objects.active = spacecraft
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Create a basic spacecraft shape
        bm = bmesh.from_mesh(spacecraft.data)
        
        # Scale and modify to look more like a spacecraft
        bmesh.ops.scale(bm, vec=(2.0, 1.0, 0.5), verts=bm.verts)
        
        # Add some details - create solar panels
        bmesh.ops.extrude_face_region(bm, faces=[f for f in bm.faces if abs(f.normal.x) > 0.9])
        bmesh.ops.scale(bm, vec=(1.0, 3.0, 0.1), verts=[v for v in bm.verts if abs(v.co.x) > 1.5])
        
        # Update mesh
        bm.to_mesh(spacecraft.data)
        bm.free()
        
        bpy.ops.object.mode_set(mode='OBJECT')
        spacecraft.scale = (scale, scale, scale)
        
        # Add materials
        self._apply_spacecraft_material(spacecraft)
        
        self.spacecraft_objects[name] = spacecraft
        return spacecraft
    
    def create_space_station(self, name: str = "SpaceStation",
                           position: Tuple[float, float, float] = (0, 0, 0),
                           scale: float = 2.0) -> bpy.types.Object:
        """Create a space station model."""
        # Create main hub
        bpy.ops.mesh.primitive_cylinder_add(location=position, radius=1.5, depth=1.0)
        station = bpy.context.active_object
        station.name = name
        
        # Enter edit mode for modifications
        bpy.context.view_layer.objects.active = station
        bpy.ops.object.mode_set(mode='EDIT')
        
        bm = bmesh.from_mesh(station.data)
        
        # Add docking modules
        for i, angle in enumerate([0, 90, 180, 270]):
            # Create docking arm
            ret = bmesh.ops.extrude_face_region(bm, faces=[f for f in bm.faces if f.normal.z > 0.9])
            extruded_verts = [v for v in ret['faces'][0].verts]
            
            # Move and scale the docking arm
            bmesh.ops.translate(bm, vec=(0, 0, 1.0), verts=extruded_verts)
            bmesh.ops.scale(bm, vec=(0.5, 0.5, 2.0), verts=extruded_verts)
        
        # Update mesh
        bm.to_mesh(station.data)
        bm.free()
        
        bpy.ops.object.mode_set(mode='OBJECT')
        station.scale = (scale, scale, scale)
        
        # Add materials
        self._apply_station_material(station)
        
        self.spacecraft_objects[name] = station
        return station
    
    def _apply_spacecraft_material(self, obj: bpy.types.Object):
        """Apply spacecraft material."""
        # Create material
        mat = bpy.data.materials.new(name=f"{obj.name}_Material")
        mat.use_nodes = True
        
        # Clear default nodes
        mat.node_tree.nodes.clear()
        
        # Add principled BSDF
        principled = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs[0].default_value = (0.8, 0.8, 0.9, 1.0)  # Base color - light gray
        principled.inputs[4].default_value = 0.9  # Metallic
        principled.inputs[7].default_value = 0.1  # Roughness
        
        # Add output
        output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(principled.outputs[0], output.inputs[0])
        
        # Assign material
        obj.data.materials.append(mat)
    
    def _apply_station_material(self, obj: bpy.types.Object):
        """Apply space station material."""
        # Create material
        mat = bpy.data.materials.new(name=f"{obj.name}_Material")
        mat.use_nodes = True
        
        # Clear default nodes
        mat.node_tree.nodes.clear()
        
        # Add principled BSDF
        principled = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs[0].default_value = (0.9, 0.9, 0.8, 1.0)  # Base color - warm white
        principled.inputs[4].default_value = 0.7  # Metallic
        principled.inputs[7].default_value = 0.2  # Roughness
        principled.inputs[17].default_value = (1.0, 0.8, 0.6)  # Emission color for lights
        principled.inputs[18].default_value = 0.5  # Emission strength
        
        # Add output
        output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(principled.outputs[0], output.inputs[0])
        
        # Assign material
        obj.data.materials.append(mat)


class DockingPortRenderer:
    """Renders docking ports and connection mechanisms."""
    
    def __init__(self):
        self.docking_ports = {}
    
    def create_docking_port(self, name: str = "DockingPort",
                          position: Tuple[float, float, float] = (0, 0, 0),
                          normal: Tuple[float, float, float] = (0, 0, 1)) -> bpy.types.Object:
        """Create a docking port."""
        # Create basic cylinder for docking port
        bpy.ops.mesh.primitive_cylinder_add(location=position, radius=0.3, depth=0.2)
        port = bpy.context.active_object
        port.name = name
        
        # Orient according to normal
        normal_vec = mathutils.Vector(normal)
        port.rotation_euler = normal_vec.to_track_quat('Z', 'Y').to_euler()
        
        # Add docking port material
        self._apply_docking_port_material(port)
        
        self.docking_ports[name] = port
        return port
    
    def create_docking_mechanism(self, spacecraft_obj: bpy.types.Object,
                               port_position: Tuple[float, float, float]) -> bpy.types.Object:
        """Create docking mechanism attached to spacecraft."""
        # Create extendable docking probe
        bpy.ops.mesh.primitive_cylinder_add(
            location=port_position, 
            radius=0.1, 
            depth=1.0
        )
        mechanism = bpy.context.active_object
        mechanism.name = f"{spacecraft_obj.name}_DockingMechanism"
        
        # Parent to spacecraft
        mechanism.parent = spacecraft_obj
        
        # Add material
        self._apply_mechanism_material(mechanism)
        
        return mechanism
    
    def _apply_docking_port_material(self, obj: bpy.types.Object):
        """Apply docking port material."""
        mat = bpy.data.materials.new(name=f"{obj.name}_Material")
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        
        principled = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs[0].default_value = (1.0, 0.6, 0.2, 1.0)  # Orange color
        principled.inputs[4].default_value = 0.8  # Metallic
        principled.inputs[7].default_value = 0.3  # Roughness
        
        output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(principled.outputs[0], output.inputs[0])
        
        obj.data.materials.append(mat)
    
    def _apply_mechanism_material(self, obj: bpy.types.Object):
        """Apply docking mechanism material."""
        mat = bpy.data.materials.new(name=f"{obj.name}_Material")
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        
        principled = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs[0].default_value = (0.2, 0.8, 1.0, 1.0)  # Blue color
        principled.inputs[4].default_value = 0.9  # Metallic
        principled.inputs[7].default_value = 0.1  # Roughness
        
        output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(principled.outputs[0], output.inputs[0])
        
        obj.data.materials.append(mat)
