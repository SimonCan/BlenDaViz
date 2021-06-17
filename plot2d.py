# plot2d.py
"""
Contains routines to two-dimensional plots.

Created on Wed Oct 31 20:30:00 2018

@author: Simon Candelaresi
"""


'''
Test:
import sys
sys.path.append('~/codes/blendaviz')
import numpy as np
import importlib
import blendaviz as blt
importlib.reload(blt.plot2d)
importlib.reload(blt)

x0 = np.linspace(-3, 3, 20)
y0 = np.linspace(-3, 3, 20)
time = np.linspace(0, 100, 101)
x, y, tt = np.meshgrid(x0, y0, time, indexing='ij')
z = np.ones_like(x)
alpha = 0.5
z = (1 - x**2-y**2)*np.exp(-(x**2+y**2)/5)
z = z*np.sin(tt/30)

mesh = blt.mesh(x, y, z, c='r', time=time, alpha=alpha)
mesh.plot()
'''

def mesh(x, y, z=None, c=None, alpha=None, vmax=None, vmin=None, color_map=None,
         time=None):
    """
    Plot a 2d surface with optional color.

    call signature:

    mesh(x, y, z=None, c=None, alpha=None, vmax=None, vmin=None, color_map=None,
         time=None)

    Keyword arguments:

    *x, y, z*:
      x, y and z coordinates of the points on the surface of shape (nu, nv)
      or of shape (nu, nv, nt) for time dependent arrays.
      If z == None then z = 0.

    *c*:
      Values to be used for the colors.
      Can be a character or string for constant color, e.g. 'red'
      or array of shape (nu, nv) for time independent colors
      or array of shape (nu, nv, nt) for time dependent colors.

    *alpha*:
      Alpha values defining the opacity.
      Single float or 2d array of shape (nu, nv).

    *vmin, vmax*
      Minimum and maximum values for the colormap.
      If not specified, determine from the input arrays.
      Can be float or array of length nt.

    *color_map*:
      Color map for the values stored in the array 'c'.
      These are the same as in matplotlib.

    *time*:
      Float array with the time information of the data.
      Has length nt.

    Examples:
      import numpy as np
      import blendaviz as blt
      x0 = np.linspace(-3, 3, 20)
      y0 = np.linspace(-3, 3, 20)
      x, y = np.meshgrid(x0, y0, indexing='ij')
      z = np.ones_like(x)*np.linspace(0, 2, 20)
      alpha = 0.5
      z = (1 - x**2-y**2)*np.exp(-(x**2+y**2)/5)
      m = blt.mesh(x, y, z, c='r', alpha=alpha)
      m.c = z
      m.plot()
      m.z = None
      m.plot()
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

        import bpy
        import blendaviz as blt

        # Define the members that can be seen by the user.
        self.x = 0
        self.y = 0
        self.z = None
        self.c = None
        self.alpha = None
        self.vmin = None
        self.vmax = None
        self.time = None
        self.time_index = 0
        self.color_map = None
        self.mesh_data = None
        self.mesh_object = None
        self.mesh_material = None
        self.deletable_object = None

        # Define the locally used time-independent data and parameters.
        self._x = 0
        self._y = 0
        self._z = None
        self._c = None
        self._vmin = None
        self._vmax = None

        # Set the handler function for frame changes (time).
        bpy.app.handlers.frame_change_pre.append(self.time_handler)

        # Add the plot to the stack.
        blt.__stack__.append(self)


    def plot(self):
        """
        Plot the 2d mesh.
        """

        import bpy
        import numpy as np
        import matplotlib.cm as cm

        # Check if there is any time array.
        if not self.time is None:
            if not isinstance(self.time, np.ndarray):
                print("Error: time is not a valid array.")
                return -1
            elif self.time.ndim != 1:
                print("Error: time array must be 1d.")
                return -1
            # Determine the time index.
            self.time_index = np.argmin(abs(bpy.context.scene.frame_float - self.time))
        else:
            self.time = np.array([0])
            self.time_index = 0

        # Check the validity of the input arrays.
        if not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray):
            print("Error: x OR y array invalid.")
            return -1
        if not isinstance(self.z, np.ndarray) and not isinstance(self.c, np.ndarray):
            print("Error: either z or c or both must be arrays.")
            return -1
        if isinstance(self.z, np.ndarray):
            if not self.z.shape[:2] == self.x.shape[:2]:
                print("Error: z array shape invalid.")
                return -1
        if isinstance(self.c, np.ndarray):
            if not self.c.shape[:2] == self.x.shape[:2]:
                print("Error: c array shape invalid.")
                return -1
        if not self.alpha:
            self.alpha = 1
        if isinstance(self.alpha, np.ndarray):
            if self.alpha.shape != (1, ):
                if not self.alpha.shape[:2] == self.x.shape[:2]:
                    print("Error: alpha array shape invalid.")
                    return -1
        else:
            self.alpha = np.array([self.alpha])

        # Point the local variables to the correct arrays.
        if self.x.ndim == 3:
            self._x = self.x[:, :, self.time_index]
        else:
            self._x = self.x
        if self.y.ndim == 3:
            self._y = self.y[:, :, self.time_index]
        else:
            self._y = self.y
        if not isinstance(self.z, np.ndarray):
            self._z = np.zeros_like(self._x)
        else:
            if self.z.ndim == 3:
                self._z = self.z[:, :, self.time_index]
            else:
                self._z = self.z
        if not (self.x.shape == self.y.shape == self.z.shape):
            print("Error: x, y, z array shapes invalid.")
            return -1
        if isinstance(self.vmin, np.ndarray):
            self._vmin = self.vmin[self.time_index]
        else:
            self._vmin = self.vmin
        if isinstance(self.vmax, np.ndarray):
            self._vmax = self.vmax[self.time_index]
        else:
            self._vmax = self.vmax
        # Set the array to be used for the color map.
        if not isinstance(self.c, np.ndarray):
            if self.c is None:
                self._c = self._z
            else:
                self._c = self.c
        else:
            if self.c.ndim == 3:
                self._c = self.c[:, :, self.time_index]
            else:
                self._c = self.c
        
        # Delete existing meshes.
        if not self.mesh_object is None:
            bpy.ops.object.select_all(action='DESELECT')
            self.mesh_object.select_set(state=True)
            bpy.ops.object.delete()
            self.mesh_object = None

        # Delete existing materials.
        if not self.mesh_material is None:
            bpy.data.materials.remove(self.mesh_material)

        # Create the vertices from the data.
        vertices = []
        for idx in range(self._x.shape[0]*self._x.shape[1]):
            vertices.append((self._x[:, :].flatten()[idx],
                             self._y[:, :].flatten()[idx],
                             self._z[:, :].flatten()[idx]))

        # Create the faces from the data.
        faces = []
        count = 0
        for idx in range((self._x.shape[0]-1)*(self._x.shape[1])):
            if count < self._x.shape[1]-1:
                faces.append((idx, idx+1, (idx+self._x.shape[1])+1, (idx+self._x.shape[1])))
                count += 1
            else:
                count = 0

        # Create mesh and object.
        self.mesh_data = bpy.data.meshes.new("DataMesh")
        self.mesh_object = bpy.data.objects.new("ObjMesh", self.mesh_data)

        # Create mesh from the given data.
        self.mesh_data.from_pydata(vertices, [], faces)
        self.mesh_data.update(calc_edges=True)

        # Assign a material to the surface.
        self.mesh_material = bpy.data.materials.new('MaterialMesh')
        self.mesh_data.materials.append(self.mesh_material)

        # Create the texture.
        if isinstance(self._c, np.ndarray):
            mesh_image = bpy.data.images.new('ImageMesh', self._c.shape[0], self._c.shape[1])
            pixels = np.array(mesh_image.pixels)
            
            # Determine the minimum and maximum value for the color map.
            vmin = self._vmin
            vmax = self._vmax
            if self._vmin is None:
                vmin = np.min(self._c)
            if self._vmax is None:
                vmax = np.max(self._c)

            # Assign the RGBa values to the pixels.
            if self.color_map is None:
                self.color_map = cm.viridis
            pixels[0::4] = self.color_map((self._c.flatten() - vmin)/(vmax - vmin))[:, 0]
            pixels[1::4] = self.color_map((self._c.flatten() - vmin)/(vmax - vmin))[:, 1]
            pixels[2::4] = self.color_map((self._c.flatten() - vmin)/(vmax - vmin))[:, 2]
            pixels[3::4] = self.alpha.flatten()
            mesh_image.pixels[:] = np.swapaxes(pixels.reshape([self._x.shape[0],
                                                               self._x.shape[1], 4]), 0, 1).flatten()[:]

            # Assign the texture to the material.
            self.mesh_material.use_nodes = True
            self.mesh_texture = self.mesh_material.node_tree.nodes.new('ShaderNodeTexImage')
            self.mesh_texture.image = mesh_image
            links = self.mesh_material.node_tree.links
            links.new(self.mesh_texture.outputs[0],
                      self.mesh_material.node_tree.nodes.get("Principled BSDF").inputs[0])

            # Link the mesh object with the scene.
            bpy.context.scene.collection.objects.link(self.mesh_object)

            # UV mapping for the new texture.
            bpy.context.view_layer.objects.active = self.mesh_object
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0)
            bpy.ops.object.mode_set(mode='OBJECT')

            # UV mapping of the material on the mesh.
            polygon_idx = 0
            for polygon in self.mesh_object.data.polygons:
                for idx in polygon.loop_indices:
                    x_idx = polygon_idx // (self._x.shape[1] - 1)
                    y_idx = polygon_idx % (self._x.shape[1] - 1)
                    if (idx-4*polygon_idx) == 1 or (idx-4*polygon_idx) == 2:
                        y_idx += 1
                    if (idx-4*polygon_idx) == 2 or (idx-4*polygon_idx) == 3:
                        x_idx += 1
                    uv_new = np.array([(float(x_idx)+0.5)/self._x.shape[0],
                                       (float(y_idx)+0.5)/self._x.shape[1]])
                    self.mesh_object.data.uv_layers[0].data[idx].uv[0] = uv_new[0]
                    self.mesh_object.data.uv_layers[0].data[idx].uv[1] = uv_new[1]
                polygon_idx += 1
        else:
            # Transform color string into rgba.
            from blendaviz import colors

            self.mesh_material.diffuse_color = colors.string_to_rgba(self._c)

            # Link the mesh object with the scene.
            bpy.context.scene.collection.objects.link(self.mesh_object)

        # Make the mesh the deletable object.
        self.deletable_object = self.mesh_object

        return 0


    def time_handler(self, scene, depsgraph):
        """
        Function to be called whenever any Blender animation functions are used.
        Updates the plot according to the function specified.
        """

        if not self.time is None:
            self.plot()
        else:
            pass
