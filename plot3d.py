# plot3d.py
"""
Contains routines to three-dimensional plots.

Created on Wed Dec 16 10:34:00 2018

@author: Simon Candelaresi
"""

'''
Test:
import numpy as np
import importlib
import blender as blt
importlib.reload(blt.plot3d)
importlib.reload(blt)
x0 = np.linspace(-3, 3, 20)
y0 = np.linspace(-3, 3, 20)
z0 = np.linspace(-3, 3, 20)
x, y, z = np.meshgrid(x0, y0, z0, indexing='ij')
phi = np.exp(-(x**2+y**2+z**2))*np.cos(z)
alpha = np.sin(np.linspace(phi.min(), phi.max(), 100))
vol = vol.mesh(phi, x, y, z, alpha=)
vol.plot()
'''

# TODO:
# vol:
# - 1) Specify alpha values as 1xn or 2xn array.
# quiver:
# - 1) Pivot points.
# streamlines:
# - 1) Different metrics.

def vol(phi, x, y, z, alpha=None, color_map=None):
    """
    Plot a 3d volume rendering of a scalar field.

    call signature:

    vol(phi, x, y, z, alpha=None, color_map=None)

    Keyword arguments:

    *phi*:
      Scalar field in 3d.

    *x, y, z*:
      1d arrays for the coordinates.

    *alpha*:
      1d array of floats between 0 and 1 containing the alpha values for phi.
      If specified as 2d array fo shape [2, n_alpha] the value of phi
      for which the alpha is true is used. Those phis must be monotonously
      increasing.

    *color_map*:
      Color map for the values stored in the array 'c'.
      These are the same as in matplotlib.
    """

    import inspect

    # Assign parameters to the Volume objects.
    volume_return = Volume()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(volume_return, argument, argument_dict[argument])
    volume_return.plot()
    return volume_return


class Volume(object):
    """
    Volume class including the data, 3d texture and parameters.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        self.phi = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.alpha = None
        self.color_map = None


    def plot(self):
        '''
        Plot the 3d texture.
        '''

        import bpy
        import numpy as np
        import matplotlib.cm as cm

        # Check the validity of the input arrays.
        if (not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray)
            or not isinstance(self.z, np.ndarray)):
            print("Error: x OR y OR z array invalid.")
            return -1
        if not isinstance(self.phi, np.ndarray):
            print("Error: phi must be array.")
            return -1
        if not isinstance(self.alpha, np.ndarray):
            print("Error: alpha must be array.")
            return -1

        # Using volumetric textures or voxels?

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
            pixels[3::4] = 1
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
