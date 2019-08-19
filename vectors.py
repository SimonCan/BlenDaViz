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
myvec = bz.vec((1,0,0), (0,1,0), 1, color='k')
myvec.change_loc((1,1,1))

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

import BlenDaViz as bz
import numpy as np
import integrate as int
import matplotlib.cm as cm
cm = cm.coolwarm
for x in np.linspace(-1.5,1.5,15):
    for y in np.linspace(-1.5, 1.5, 15):
        loc = (x, y, 0)
        dir = -int.BHopf(np.array(loc))
        color = cm(dir[2]/(-1*int.BHopf(np.array((0,0,0)))[2]))[:3]
        print(color)
        bz.vec(loc, dir, length=.5, color=color)

'''

# TODO:
# + 1) Make root mesh invisible
# - 2) Add color.
# - 3) Check user input.
# - 4) Add documentation.

def vec(root_point, direction, length, color=None, thin=None):
    """
    add a 3d vector mesh at a location in your scene

    Keyword arguments:
    *root_point*:
      the 'anchor' of the arrow, where the base point lies

    *direction*
      the direction of the arrow. If not normalized, it will be.

    *length*:
      the length of the arrow

    *color*:
      The color of the arrow

    *thin*:
        thinning factor, will make the arrow this much thinner
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
        #deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        if not isinstance(self.root_point, tuple) and not isinstance(self.root_point, np.ndarray):
            print("Error: root_point not a tuple of numbers or ndarray.")
            return -1


        if not self.mesh_object is None:
            bpy.ops.object.select_all(action='DESELECT') # don't you always need to deselect?
            self.mesh_object.select_set(state = True)
            bpy.ops.object.delete()
            self.mesh_object = None

        # Delete existing materials.
        if not self.mesh_material is None:
            bpy.data.materials.remove(self.mesh_material)

        if bpy.data.objects.get("arrow_Mesh") is None:
            #load the invisible arrow
            print('reading arrow from file') #diagnostic, REMOVE
            arrowpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'arrow.obj') # find arrow.obj in same folder as this file.
            bpy.ops.import_scene.obj(filepath=arrowpath)
            bpy.context.scene.collection.objects.unlink(bpy.data.objects['arrow_Mesh']) #unlink so this arrow is invisible



        self.mesh_data = bpy.data.objects['arrow_Mesh'].data.copy()
        self.mesh_object = bpy.data.objects.new("vector", self.mesh_data )
        bpy.context.scene.collection.objects.link(self.mesh_object)#must link before you select!
        self.mesh_object.select_set(state = True)

        # Assign a material to the surface.
        self.mesh_material = bpy.data.materials.new('MaterialMesh')
        self.mesh_data.materials.append(self.mesh_material) # inherits previous material, changing this probably changes all?


        # Make root_point and direction mathutil.Vector type vectors
        self.root_point = Vector(self.root_point)
        self.direction = Vector(self.direction)
        self.direction.normalize()

        if self.thin is not None:
            #print('thinning...')
            bpy.ops.transform.resize(value=(self.thin, 1, 1))
            bpy.ops.transform.resize(value=(1/self.thin, 1/self.thin, 1/self.thin))
        # rotate using Quaternions! Why Quaternions? Because awesome!
        self.mesh_object.location = self.root_point
        self.mesh_object.rotation_mode = 'QUATERNION'
        self.mesh_object.rotation_quaternion = self.direction.to_track_quat('X','Z')




        color_rgba = self.color
        if isinstance(self.color, str):
            if self.color == 'random':
                from numpy.random import rand
                color_rgba=(rand(), rand(), rand(), 1)
            else:
                from . import colors
                color_rgba = colors.string_to_rgb(self.color)
        self.mesh_material.diffuse_color = color_rgba
        self.mesh_object.active_material = self.mesh_material

        bpy.ops.transform.resize(value=(self.length, self.length, self.length))



    def change_loc(self, newroot):
        """
        updates the location of an existing vector.
        *root_point*:
            The location where the vector is 'rooted'
        """
        import numpy as np
        from mathutils import Vector
        if not isinstance(newroot, tuple) and not isinstance(newroot, np.ndarray):
            print("Error: root_point not a tuple of numbers or ndarray.")
            return -1
        self.root_point = Vector(newroot)
        self.mesh_object.location = self.root_point











