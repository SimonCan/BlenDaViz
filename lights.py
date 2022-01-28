# lights.py
"""
Contains routines to add and handle lights.
"""


def adjust_lights():
    """
    Add lights or change their parameters.

    Signature:

    adjust_lights()
    """

    import bpy
    import blendaviz as blt
    import numpy as np

    for i in range(6):
        if blt.house_keeping.lights[i] is None:
            bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 0))
            blt.house_keeping.lights[i] = bpy.context.object
            blt.house_keeping.lights[i].data.energy = 1.5
    blt.house_keeping.lights[0].location = [blt.house_keeping.x_max + (blt.house_keeping.x_max - blt.house_keeping.x_min)/4,
                                            (blt.house_keeping.y_min + blt.house_keeping.y_max)/2,
                                            (blt.house_keeping.z_min + blt.house_keeping.z_max)/2]
    blt.house_keeping.lights[1].location = [blt.house_keeping.x_min - (blt.house_keeping.x_max - blt.house_keeping.x_min)/4,
                                            (blt.house_keeping.y_min + blt.house_keeping.y_max)/2,
                                            (blt.house_keeping.z_min + blt.house_keeping.z_max)/2]
    blt.house_keeping.lights[2].location = [(blt.house_keeping.x_min + blt.house_keeping.x_max)/2,
                                            blt.house_keeping.y_max + (blt.house_keeping.y_max - blt.house_keeping.y_min)/4,
                                            (blt.house_keeping.z_min + blt.house_keeping.z_max)/2]
    blt.house_keeping.lights[3].location = [(blt.house_keeping.x_min + blt.house_keeping.x_max)/2,
                                            blt.house_keeping.y_min - (blt.house_keeping.y_max - blt.house_keeping.y_min)/4,
                                            (blt.house_keeping.z_min + blt.house_keeping.z_max)/2]
    blt.house_keeping.lights[4].location = [(blt.house_keeping.x_min + blt.house_keeping.x_max)/2,
                                            (blt.house_keeping.y_min + blt.house_keeping.y_max)/2,
                                            blt.house_keeping.z_max + (blt.house_keeping.z_max - blt.house_keeping.z_min)/4]
    blt.house_keeping.lights[5].location = [(blt.house_keeping.x_min + blt.house_keeping.x_max)/2,
                                            (blt.house_keeping.y_min + blt.house_keeping.y_max)/2,
                                            blt.house_keeping.z_min - (blt.house_keeping.z_max - blt.house_keeping.z_min)/4]
#    for light in blt.house_keeping.lights:
#        light.data.energy = 1.5
    blt.house_keeping.lights[0].rotation_euler[1] = np.pi/2
    blt.house_keeping.lights[1].rotation_euler[1] = -np.pi/2
    blt.house_keeping.lights[2].rotation_euler[0] = -np.pi/2
    blt.house_keeping.lights[3].rotation_euler[0] = np.pi/2
    blt.house_keeping.lights[4].rotation_euler[0] = 0
    blt.house_keeping.lights[5].rotation_euler[0] = np.pi

