# materials.py
"""
Contains utilities for creating and managing Blender materials.
"""

from typing import Tuple, Optional, Any


def create_material_with_color(
    name: str,
    color_rgba: Tuple[float, float, float, float],
    emission: Optional[float] = None,
    roughness: float = 1.0
) -> Any:  # bpy.types.Material
    """
    Create a Blender material with specified color and properties.

    Parameters
    ----------
    name : str
        Name of the material.
    color_rgba : Tuple[float, float, float, float]
        RGBA color values (0-1 range).
    emission : Optional[float]
        Emission strength. If None, use roughness-based material.
    roughness : float
        Surface roughness (0-1 range).

    Returns
    -------
    Material object
    """
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes

    # Find the material output node
    output_node = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output_node = node
            break

    # Remove default nodes except output
    for node in list(nodes):
        if node != output_node:
            nodes.remove(node)

    if emission is not None:
        # Create emission material
        node_emission = nodes.new(type='ShaderNodeEmission')
        node_tree.links.new(node_emission.outputs['Emission'],
                          output_node.inputs['Surface'])
        node_emission.inputs['Color'].default_value = color_rgba
        node_emission.inputs['Strength'].default_value = emission
    else:
        # Create principled BSDF material
        node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_tree.links.new(node_bsdf.outputs['BSDF'],
                          output_node.inputs['Surface'])
        node_bsdf.inputs['Base Color'].default_value = color_rgba
        node_bsdf.inputs['Roughness'].default_value = roughness

    return material


def create_textured_material(
    name: str,
    image: Any,  # bpy.types.Image
    emission: Optional[float] = None,
    roughness: float = 1.0
) -> Any:  # bpy.types.Material
    """
    Create a Blender material with an image texture.

    Parameters
    ----------
    name : str
        Name of the material.
    image : bpy.types.Image
        Image to use as texture.
    emission : Optional[float]
        Emission strength. If None, use principled BSDF.
    roughness : float
        Surface roughness (0-1 range).

    Returns
    -------
    Material object
    """
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes

    # Find the material output node
    output_node = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output_node = node
            break

    # Remove default nodes except output
    for node in list(nodes):
        if node != output_node:
            nodes.remove(node)

    # Create texture node
    texture_node = nodes.new('ShaderNodeTexImage')
    texture_node.image = image

    if emission is not None:
        # Create emission material
        node_emission = nodes.new(type='ShaderNodeEmission')
        node_tree.links.new(texture_node.outputs['Color'],
                          node_emission.inputs['Color'])
        node_tree.links.new(node_emission.outputs['Emission'],
                          output_node.inputs['Surface'])
        node_emission.inputs['Strength'].default_value = emission
    else:
        # Create principled BSDF material
        node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_tree.links.new(texture_node.outputs['Color'],
                          node_bsdf.inputs['Base Color'])
        node_tree.links.new(node_bsdf.outputs['BSDF'],
                          output_node.inputs['Surface'])
        node_bsdf.inputs['Roughness'].default_value = roughness

    return material
