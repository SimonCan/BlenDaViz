# plot2d.py
"""
Contains routines to two-dimensional plots.

Created on Wed Oct 31 20:30:00 2018

@author: Simon Candelaresi
"""

'''
Test:
import sys
sys.path.insert(0, '/home/iomsn/python')

import numpy as np
import importlib
import blender as blt
importlib.reload(blt.plot2d)
importlib.reload(blt)

x0 = np.linspace(-3, 3, 20)
y0 = np.linspace(-3, 3, 20)
x, y = np.meshgrid(x0, y0, indexing='ij')
z = np.ones_like(x)*np.linspace(0, 2, 20)
alpha = 0.5
# z = (1 - x**2-y**2)*np.exp(-(x**2+y**2)/5)

m = blt.mesh(x, y, z, c='r', alpha=alpha)
m.plot()
'''

# TODO:
# + 1) Generate 2d mesh.
# + 2) Add color.
# + 3) Check user input.
# - 4) Add documentation.
# - 5) alpha as float or array.
# + 99) Code analysis.

def mesh(x, y, z=None, c=None, alpha=None, color_map=None):
    """
    Plot a 2d surface with optional color.

    call signature:

    mesh(x, y, z, c=None, color_map=None)

    Keyword arguments:

    *x, y, z*:
      x, y and z coordinates of the points on the surface of shape [nu, nv].
      If z == None plot a planar surface.

    *c*:
      Values to be used for the colors.

    *alpha*:
      Alpha values defining the opacity.
      Single float or 2d array of shape [nu, nv].

    *color_map*:
      Color map for the values stored in the array 'c'.
      These are the same as in matplotlib.
    """

    import inspect

    # Assign parameters to the Mesh objects.
    surface_return = Surface()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(surface_return, argument, argument_dict[argument])
    surface_return.plot()
    return surface_return


class Surface(object):
    """
    Surface class including the vertices, surfaces, parameters and plotting function.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        self.x = 0
        self.y = 0
        self.z = None
        self.c = None
        self.alpha = None
        self.color_map = None
        self.mesh_data = None
        self.mesh_object = None
        self.mesh_material = None


    def plot(self):
        '''
        Plot the 2d mesh.
        '''

        import bpy
        import numpy as np
        import matplotlib.cm as cm

        # Check the validity of the input arrays.
        if not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray):
            print("Error: x OR y array invalid.")
            return -1
        if not isinstance(self.z, np.ndarray) and not isinstance(self.c, np.ndarray):
            print("Error: z OR c must be arrays.")
            return -1
        if not isinstance(self.c, np.ndarray):
            if self.c is None:
                self.c = self.z
        if not isinstance(self.z, np.ndarray):
            self.z = np.zeros_like(self.x)
        if not (self.x.shape == self.y.shape == self.z.shape):
            print("Error: array shapes invalid.")
            return -1
        if isinstance(self.c, np.ndarray):
            if not self.c.shape == self.x.shape:
                print("Error: c array shape invalid.")
                return -1
        if isinstance(self.alpha, np.ndarray):
            if not self.alpha.shape == self.x.shape:
                print("Error: alpha array shape invalid.")
                return -1
        else:
            self.alpha = np.array([self.alpha])

        # Delete existing meshes.
        if not self.mesh_object is None:
            bpy.ops.object.select_all(action='DESELECT')
            self.mesh_object.select = True
            bpy.ops.object.delete()
            self.mesh_object = None

        # Delete existing materials.
        if not self.mesh_material is None:
            bpy.data.materials.remove(self.mesh_material)

        # Create the vertices from the data.
        vertices = []
        for idx in range(self.x.size):
            vertices.append((self.x.flatten()[idx], self.y.flatten()[idx], self.z.flatten()[idx]))

        # Create the faces from the data.
        faces = []
        count = 0
        for idx in range((self.x.shape[0]-1)*(self.x.shape[1])):
            if count < self.x.shape[1]-1:
                faces.append((idx, idx+1, (idx+self.x.shape[1])+1, (idx+self.x.shape[1])))
                count += 1
            else:
                count = 0

        # Create mesh and object.
        self.mesh_data = bpy.data.meshes.new("DataMesh")
        self.mesh_object = bpy.data.objects.new("ObjMesh", self.mesh_data)
        self.mesh_object.select = True

        # Create mesh from the given data.
        self.mesh_data.from_pydata(vertices, [], faces)
        self.mesh_data.update(calc_edges=True)

        # Assign a material to the surface.
        self.mesh_material = bpy.data.materials.new('MaterialMesh')
        self.mesh_data.materials.append(self.mesh_material)

        # Create the texture.
        if isinstance(self.c, np.ndarray):
            mesh_image = bpy.data.images.new('ImageMesh', self.c.shape[0], self.c.shape[1])
            pixels = np.array(mesh_image.pixels)
            c_max = np.max(self.c)
            c_min = np.min(self.c)

            # Assign the RGBa values to the pixels.
            if self.color_map is None:
                self.color_map = cm.viridis
            pixels[0::4] = self.color_map((self.c.flatten() - c_min)/(c_max - c_min))[:, 0]
            pixels[1::4] = self.color_map((self.c.flatten() - c_min)/(c_max - c_min))[:, 1]
            pixels[2::4] = self.color_map((self.c.flatten() - c_min)/(c_max - c_min))[:, 2]
            pixels[3::4] = self.alpha.flatten()
            mesh_image.pixels[:] = np.swapaxes(pixels.reshape([self.x.shape[0],
                                                               self.x.shape[1], 4]), 0, 1).flatten()[:]

            # Assign the texture to the material.
            self.mesh_material.use_nodes = True
            self.mesh_texture = self.mesh_material.node_tree.nodes.new('ShaderNodeTexImage')
            self.mesh_texture.image = mesh_image
            links = self.mesh_material.node_tree.links
            link = links.new(self.mesh_texture.outputs[0],
                             self.mesh_material.node_tree.nodes.get("Diffuse BSDF").inputs[0])

            # Link the mesh object with the scene.
            bpy.context.scene.objects.link(self.mesh_object)

            # UV mapping for the new texture.
            bpy.context.scene.objects.active = self.mesh_object
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0)
            bpy.ops.object.mode_set(mode='OBJECT')

            # UV mapping of the material on the mesh.
            polygon_idx = 0
            for polygon in self.mesh_object.data.polygons:
                for idx in polygon.loop_indices:
                    x_idx = polygon_idx // (self.x.shape[1] - 1)
                    y_idx = polygon_idx % (self.x.shape[1] - 1)
                    if (idx-4*polygon_idx) == 1 or (idx-4*polygon_idx) == 2:
                        y_idx += 1
                    if (idx-4*polygon_idx) == 2 or (idx-4*polygon_idx) == 3:
                        x_idx += 1
                    uv_new = np.array([(float(x_idx)+0.5)/self.x.shape[0],
                                       (float(y_idx)+0.5)/self.x.shape[1]])
                    self.mesh_object.data.uv_layers[0].data[idx].uv[0] = uv_new[0]
                    self.mesh_object.data.uv_layers[0].data[idx].uv[1] = uv_new[1]
                polygon_idx += 1
        else:
            # Transform color string into rgb.
            from . import colors

            print(colors.string_to_rgb(self.c))
            self.mesh_material.diffuse_color = colors.string_to_rgb(self.c)

            # Link the mesh object with the scene.
            bpy.context.scene.objects.link(self.mesh_object)

        return 0
