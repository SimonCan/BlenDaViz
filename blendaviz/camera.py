# camera.py
"""
Contains routines to add and handle the camera.
"""


def adjust_camera() -> None:
    """
    Add a camera or change its position and focus.

    Signature:

    adjust_camera()
    """

    import bpy
    import blendaviz as blt
    import numpy as np

    # Find the center of the plot.
    x_center = (blt.house_keeping.x_max + blt.house_keeping.x_min)/2
    y_center = (blt.house_keeping.y_max + blt.house_keeping.y_min)/2
    z_center = (blt.house_keeping.z_max + blt.house_keeping.z_min)/2

    # Find the size of the plot.
    x_size = (blt.house_keeping.x_max - blt.house_keeping.x_min)
    y_size = (blt.house_keeping.y_max - blt.house_keeping.y_min)
    z_size = (blt.house_keeping.z_max - blt.house_keeping.z_min)

    # Find where to position the camera.
    x_pos = x_center + np.max([x_size, y_size, z_size])
    y_pos = y_center + np.max([x_size, y_size, z_size])
    z_pos = z_center + np.max([x_size, y_size, z_size])

    # Create the camera if it does not exist.
    if blt.house_keeping.camera is None:
        bpy.ops.object.camera_add()
        blt.house_keeping.camera = bpy.context.object

    # Position the camera.
    blt.house_keeping.camera.location = [x_pos, y_pos, z_pos]
    blt.house_keeping.camera.rotation_euler[0] = 55.5/180*np.pi
    blt.house_keeping.camera.rotation_euler[2] = 135/180*np.pi
