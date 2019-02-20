# vectors.py
"""
Contains routines to add vectors (arrows).

Tried to stay as true as possible to Simon's BlenDaViz way of
doing things.

Personally not a big fan of using argument_dict to store function arguments
as class attributes.

Created on Tue Feb 19 2019

@author: Chris Smiet (csmiet@pppl.gov)
"""

'''
Test:
import BlenDaViz as bz
bz.vec((1,0,0), (0,1,0), 1, color='k')

import BlenDaViz as bz
import numpy as np
import integrate as int
for x in np.linspace(-1.5,1.5,15):
    for y in np.linspace(-1.5, 1.5, 15):
        loc = (x, y, 0)
        dir = int.BHopf(np.array(loc))
        bz.vec(loc, dir, length=.5, color='r')

'''

# TODO:
# + 1) Make root mesh invisible
# - 2) Add color.
# - 3) Check user input.
# - 4) Add documentation.

def vec(root_point, direction, length, alpha=None, color=None ):
    """
    add a 3d vector mesh at a location in your scene

    Keyword arguments:
    *root_point*:
      the 'anchor' of the arrow, where the base point lies

    *direction*
      the direction of the arrow. If not normalized, it will be.

    *length*:
      the length of the arrow

    *alpha*:
      opacity

    *color*:
      The color of the arrow
    """
    import inspect

    # Assign parameters to the arrow objects.
    arrow_return = arrow()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(arrow_return, argument, argument_dict[argument])
    arrow_return.place()
    return arrow_return

class arrow(object):
    """
    Arrow class that adds a 3D 'vector' arrow
    """

    def __init__(self):
        self.root_point = (0,0,0) #  The point where the base of the arrow will lie
        self.direction = (0,0,0) # the direction vector
        self.length = 1
        self.alpha = None
        self.color = None
        self.mesh_data = None
        self.mesh_material = None
        self.mesh_object = None

    def place(self):
        import os
        import bpy
        import numpy as np
        from mathutils import Vector
        import matplotlib.cm as cm
        #checks on input
        #deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        if not isinstance(self.root_point, tuple):
            print("Error: root_point not a tuple of numbers.")
            return -1


        if not self.mesh_object is None:
            bpy.ops.obj.select_all(action='DESELECT') # don't you always need to deselect?
            self.mesh_object.select = True
            bpy.ops.obj.delete()
            self.mesh_object = None

        # Delete existing materials.
        if not self.mesh_material is None:
            bpy.data.materials.remove(self.mesh_material)

        if bpy.data.objects.get("arrow_Mesh") is None:
            #load the invisible arrow
            print('reading arrow from file') #diagnostic, REMOVE
            arrowpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'arrow.obj')
            bpy.ops.import_scene.obj(filepath=arrowpath)
            bpy.context.scene.objects.unlink(bpy.data.objects['arrow_Mesh']) #mesh still accessible?



        self.mesh_object = bpy.data.objects.new("vector", bpy.data.objects['arrow_Mesh'].data)
        self.mesh_data = self.mesh_object.data
        self.mesh_object.select = True

        # Assign a material to the surface.
        self.mesh_material = bpy.data.materials.new('MaterialMesh')
        self.mesh_data.materials.append(self.mesh_material)


        # Make root_point and direction mathutil.Vector type vectors
        self.root_point = Vector(self.root_point)
        self.direction = Vector(self.direction)
        self.direction.normalize()

        # rotate using Quaternions! Why Quaternions? Because awesome!
        self.mesh_object.location = self.root_point
        self.mesh_object.rotation_mode = 'QUATERNION'
        self.mesh_object.rotation_quaternion = self.direction.to_track_quat('X','Z')




        color_rgb = self.color
        if isinstance(self.color, str):
            if self.color == 'random':
                from numpy.random import rand
                color_rgb=(rand(), rand(), rand())
            else:
                from . import colors
                color_rgb = colors.string_to_rgb(self.color)
        print(color_rgb)
        self.mesh_material.diffuse_color = color_rgb
        self.mesh_object.active_material = self.mesh_material



        bpy.context.scene.objects.link(self.mesh_object)
        bpy.ops.transform.resize(value=(self.length, self.length, self.length))












