# markers.py
"""
Contains utilities for creating Blender marker primitives.
"""

from typing import Tuple, Callable, Dict, Any
import numpy as np
import numpy.typing as npt


# Marker factory dictionary mapping marker names to their creation functions
MARKER_FACTORIES: Dict[str, Callable] = {}


def _register_marker(name: str) -> Callable:
    """Decorator to register marker creation functions."""
    def decorator(func: Callable) -> Callable:
        MARKER_FACTORIES[name] = func
        return func
    return decorator


@_register_marker('cone')
def _create_cone(location: Tuple[float, float, float],
                radius: float,
                rotation: Tuple[float, float, float]) -> Any:
    """Create a cone marker."""
    import bpy
    bpy.ops.mesh.primitive_cone_add(
        location=location,
        radius1=radius,
        depth=2*radius,
        rotation=rotation
    )
    return bpy.context.object


@_register_marker('cube')
def _create_cube(location: Tuple[float, float, float],
                radius: float,
                rotation: Tuple[float, float, float]) -> Any:
    """Create a cube marker."""
    import bpy
    bpy.ops.mesh.primitive_cube_add(
        location=location,
        size=radius,
        rotation=rotation
    )
    return bpy.context.object


@_register_marker('cylinder')
def _create_cylinder(location: Tuple[float, float, float],
                    radius: float,
                    rotation: Tuple[float, float, float]) -> Any:
    """Create a cylinder marker."""
    import bpy
    bpy.ops.mesh.primitive_cylinder_add(
        location=location,
        radius=radius,
        depth=2*radius,
        rotation=rotation
    )
    return bpy.context.object


@_register_marker('ico_sphere')
def _create_ico_sphere(location: Tuple[float, float, float],
                      radius: float,
                      rotation: Tuple[float, float, float]) -> Any:
    """Create an icosphere marker."""
    import bpy
    bpy.ops.mesh.primitive_ico_sphere_add(
        location=location,
        radius=radius,
        rotation=rotation
    )
    return bpy.context.object


@_register_marker('uv_sphere')
def _create_uv_sphere(location: Tuple[float, float, float],
                     radius: float,
                     rotation: Tuple[float, float, float]) -> Any:
    """Create a UV sphere marker."""
    import bpy
    bpy.ops.mesh.primitive_uv_sphere_add(
        location=location,
        radius=radius,
        rotation=rotation
    )
    return bpy.context.object


@_register_marker('torus')
def _create_torus(location: Tuple[float, float, float],
                 radius: float,
                 rotation: Tuple[float, float, float]) -> Any:
    """Create a torus marker."""
    import bpy
    bpy.ops.mesh.primitive_torus_add(
        location=location,
        major_radius=radius,
        minor_radius=radius/4,
        rotation=rotation
    )
    return bpy.context.object


@_register_marker('monkey')
def _create_monkey(location: Tuple[float, float, float],
                  radius: float,
                  rotation: Tuple[float, float, float]) -> Any:
    """Create a monkey (Suzanne) marker."""
    import bpy
    bpy.ops.mesh.primitive_monkey_add(
        location=location,
        size=radius,
        rotation=rotation
    )
    return bpy.context.object


def create_markers(
    marker_type: str,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    z: npt.NDArray[np.floating],
    radius: npt.NDArray[np.floating],
    rotation_x: npt.NDArray[np.floating],
    rotation_y: npt.NDArray[np.floating],
    rotation_z: npt.NDArray[np.floating]
) -> list:
    """
    Create multiple markers at specified locations.

    Parameters
    ----------
    marker_type : str
        Type of marker: 'cone', 'cube', 'cylinder', 'ico_sphere',
        'uv_sphere', 'torus', 'monkey'.
    x, y, z : ndarray
        Coordinates for each marker.
    radius : ndarray
        Radius/size for each marker.
    rotation_x, rotation_y, rotation_z : ndarray
        Rotation angles for each marker.

    Returns
    -------
    list
        List of created marker objects.
    """
    if marker_type not in MARKER_FACTORIES:
        raise ValueError(f"Unknown marker type: {marker_type}. "
                        f"Available types: {list(MARKER_FACTORIES.keys())}")

    factory = MARKER_FACTORIES[marker_type]
    markers = []

    for idx in range(x.shape[0]):
        location = (x[idx], y[idx], z[idx])
        rotation = (rotation_x[idx], rotation_y[idx], rotation_z[idx])
        marker = factory(location, radius[idx], rotation)
        markers.append(marker)

    return markers
