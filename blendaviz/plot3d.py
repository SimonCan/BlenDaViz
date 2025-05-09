# plot3d.py
"""
Contains routines to three-dimensional plots.
"""


from blendaviz.generic import GenericPlot


def quiver(x, y, z, u, v, w, pivot='middle', length=1,
           radius_shaft=0.25, radius_tip=0.5, scale=1,
           color=(0, 1, 0, 1), emission=None, roughness=1,
           vmin=None, vmax=None, color_map=None, time=None):
    """
    Plot arrows for a given vector field.

    Signature:

    quiver(x, y, z, u, v, w, pivot='middle', length=1,
           radius_shaft=0.25, radius_tip=0.5, scale=1,
           color=(0, 1, 0, 1), emission=None, roughness=1,
           vmin=None, vmax=None, color_map=None, time=None):

    Parameters
    ----------
    x, y, z:  x, y and z position of the data.
        These can be 1d arrays of the same length n or of shape (nx, ny, nz)
        for time independent data
        or of shape (n, nt) or (nx, ny, nz, nt) for time dependent data.

    u, v, w:  x, y and z components of the vector field.
        Must be of the same shape as the x, y and z arrays.

    pivot:  Part of the arrow around which it is rotated.
        Can be 'tail', 'mid', 'middle' or 'tip'.

    length:  Length of the arrows.
        If specified as the string 'magnitude' use the vector's magnitude.

    radius_shaft, radius_tip:  Radii of the shaft and tip of the arrows.
        If specified as string 'magnitude' use the vector's magnitude
        and multiply by 0.25 for radius_shaft and 0.5 for radius_tip.

    scale:  Scale the arrows (length, raddius_shaft and radius_tip).
        Can be constant or array of the same shape as x, y and z.

    color:  rgba values of the form (r, g, b, a) with 0 <= r, g, b, a <= 1, or string,
        e.g. 'red' or character, e.g. 'r', or list of strings/character,
        or [n, 4] array with rgba values or array of the same shape as input array
        or 'magnitude' (use vector length).

    emission:  Light emission by the arrows. This overrides 'roughness'.
        Real number or array or 'magnitude' (use vector length).

    roughness:  Texture roughness.

    vmin, vmax:  Minimum and maximum values for the colormap.
        If not specify, determine from the input arrays.

    color_map:  Color map for the values stored in the array 'c'.
        These are the same as in matplotlib.

    time:  Float array with the time information of the data.
        Has length nt.

    Returns
    -------
    3d Quiver plot object.

    Examples
    --------
    >>> import numpy as np
    >>> import blendaviz as blt
    >>> x = np.linspace(-2, 2, 5)
    >>> y = np.linspace(-2, 2, 5)
    >>> z = np.linspace(-2, 2, 5)
    >>> u = np.array([1, 0, 0, 1, 0])
    >>> v = np.array([0, 1, 0, 1, 1])
    >>> w = np.array([0, 0, 1, 0, 1])
    >>> qu = blt.quiver(x, y, z, u, v, w, pivot='mid', color='magnitude')
    """

    import inspect

    # Assign parameters to the arrow objects.
    quiver_return = Quiver3d()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(quiver_return, argument, argument_dict[argument])
    quiver_return.plot()
    return quiver_return


class Quiver3d(GenericPlot):
    """
    Quiver class containing geometry, parameters and plotting function.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        import bpy
        import blendaviz as blt

        super().__init__()

        # Define the members that can be seen by the user.
        self.x = 0
        self.y = 0
        self.z = 0
        self.u = 0
        self.v = 0
        self.w = 0
        self.pivot = 'middle'
        self.length = 1
        self.radius_shaft = 0.25
        self.radius_tip = 0.5
        self.scale = 1
        self.time = None
        self.time_index = 0
        self.color = (0, 1, 0, 1)
        self.emission = None
        self.roughness = 1
        self.vmin = None
        self.vmax = None
        self.color_map = None
        self.arrow_mesh = None
        self.mesh_material = None
        self.deletable_object = None

        # Define the locally used time-independent data and parameters.
        self._x = 0
        self._y = 0
        self._z = 0
        self._u = 0
        self._v = 0
        self._w = 0

        # Set the handler function for frame changes (time).
        bpy.app.handlers.frame_change_pre.append(self.time_handler)

        # Add the plot to the stack.
        blt.plot_stack.append(self)


    def plot(self):
        """
        Plot the arrows.
        """

        import bpy
        import numpy as np
        from mathutils import Vector
        from blendaviz import colors

        # Check if there is any time array.
        if not self.time is None:
            if not isinstance(self.time, np.ndarray):
                print("Error: time is not a valid array.")
                return -1
            if self.time.ndim != 1:
                print("Error: time array must be 1d.")
                return -1
            # Determine the time index.
            self.time_index = np.argmin(abs(bpy.context.scene.frame_float - self.time))
        else:
            self.time = np.array([0])
            self.time_index = 0

        # Check the validity of the input arrays.
        if not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray) \
           or not isinstance(self.z, np.ndarray):
            print("Error: x OR y OR z array invalid.")
            return -1
        if not isinstance(self.u, np.ndarray) or not isinstance(self.v, np.ndarray) \
           or not isinstance(self.w, np.ndarray):
            print("Error: u OR v OR w array invalid.")
            return -1
        if not (self.x.shape == self.y.shape == self.z.shape == \
                self.u.shape == self.v.shape == self.w.shape):
            print("Error: input array shapes invalid.")
            return -1

        # Point the local variables to the correct arrays.
        arrays_with_time_list = ['x', 'y', 'z', 'u', 'v', 'w']
        for array_with_time in arrays_with_time_list:
            array_value = getattr(self, array_with_time)
            if array_value.ndim in (1, 3):
                setattr(self, '_' + array_with_time, array_value.ravel())
            else:
                setattr(self, '_' + array_with_time, array_value[..., self.time_index].ravel())

        # Delete existing meshes.
        if not self.arrow_mesh is None:
            bpy.ops.object.select_all(action='DESELECT')
            if self.object_reference_valid(self.arrow_mesh):
                self.arrow_mesh.select_set(True)
                bpy.context.view_layer.objects.active = self.arrow_mesh
                bpy.ops.object.delete()
            self.arrow_mesh = None
        self.arrow_mesh = []

        # Delete existing materials.
        if not self.mesh_material is None:
            if isinstance(self.mesh_material, list):
                for mesh_material in self.mesh_material:
                    bpy.data.materials.remove(mesh_material)
            else:
                bpy.data.materials.remove(self.mesh_material)

        # Prepare the material colors.
        if isinstance(self.color, str):
            if self.color == 'magnitude':
                self.color = np.sqrt(self._u**2 + self._v**2 + self._w**2)
        if isinstance(self.color, np.ndarray):
            if self.vmin == self.vmax:
                self.vmax = 2*self.color.min()
            if (self.vmin == 0) and (self.vmax == 0):
                self.vmax = 1.0
            color_rgba = colors.make_rgba_array(self.color, self._x.shape[0],
                                                self.color_map, self.vmin, self.vmax)
        if isinstance(self.color, tuple) or isinstance(self.color, list):
            color_rgba = colors.make_rgba_array(self.color, self._x.shape[0])

        # Prepare the materials list.
        self.mesh_material = []

        # Plot the arrows.
        for idx in range(self._x.shape[0]):
            # Determine the length of the arrow.
            magnitude = np.sqrt(self._u[idx]**2 + self._v[idx]**2 + self._w[idx]**2)
            normed = np.array([self._u[idx], self._v[idx], self._w[idx]])/magnitude
            rotation = Vector((0, 0, 1)).rotation_difference([self._u[idx],
                                                              self._v[idx],
                                                              self._w[idx]]).to_euler()

            # Define the arrow's length.
            if self.length == 'magnitude':
                length = magnitude
            else:
                length = self.length
            length *= self.scale

            # Define the arrows' radii.
            radius_shaft = self.radius_shaft
            radius_shaft *= self.scale

            # Define the arrows' scale.
            radius_tip = self.radius_tip
            radius_tip *= self.scale

            if self.pivot == 'tail':
                location = [self._x[idx] + length*normed[0]/2,
                            self._y[idx] + length*normed[1]/2,
                            self._z[idx] + length*normed[2]/2]
            if self.pivot == 'tip':
                location = [self._x[idx] - length*normed[0]/2,
                            self._y[idx] - length*normed[1]/2,
                            self._z[idx] - length*normed[2]/2]
            if self.pivot in ('mid', 'middle'):
                location = [self._x[idx], self._y[idx], self._z[idx]]
            location = np.array(location)

            # Construct the arrow using a cylinder and cone.
            bpy.ops.mesh.primitive_cylinder_add(radius=radius_shaft, depth=length/2,
                                                location=location-normed*length/4,
                                                rotation=rotation)
            self.arrow_mesh.append(bpy.context.object)
            bpy.ops.mesh.primitive_cone_add(radius1=radius_tip, radius2=0, depth=length/2,
                                            location=location+normed*length/4, rotation=rotation)
            self.arrow_mesh.append(bpy.context.object)

            self.__set_material(idx, color_rgba)

        # Group the meshes together.
        for mesh in self.arrow_mesh[::-1]:
            mesh.select_set(state=True)
        bpy.ops.object.join()
        self.arrow_mesh = bpy.context.object
        self.arrow_mesh.select_set(state=False)

        # Make the mesh the deletable object.
        self.deletable_object = self.arrow_mesh

        self.update_globals()

        return 0


    def __set_material(self, idx, color_rgba):
        """
        Set the mesh material.

        Signature:

        __set_material(idx, color_rgba):

        Parameters
        ----------
        idx:  Index of the material.

        color_rgba:  The rgba values of the colors to be used.
        """

        import bpy
        import numpy as np

        # Deterimne if we need a list of materials, i.e. for every arrow mesh one.
        if any([isinstance(self.color, np.ndarray),
                isinstance(self.emission, np.ndarray),
                isinstance(self.roughness, np.ndarray),
                isinstance(self.color, list),
                isinstance(self.emission, list),
                isinstance(self.roughness, list)]):
            list_material = True
        else:
            list_material = False

        # Transform single values to arrays.
        if list_material:
            if color_rgba.shape[0] != self._x.shape[0]:
                color_rgba = np.repeat(color_rgba, self._x.shape[0], axis=0)
            if not isinstance(self.roughness, np.ndarray):
                self.roughness = np.ones(self._x.shape[0])*self.roughness
            if not self.emission is None:
                if not isinstance(self.emission, np.ndarray):
                    self.emission = np.ones(self.emission[0])*self.emission

        # Set the material.
        if list_material:
            self.mesh_material.append(bpy.data.materials.new('material'))
            self.arrow_mesh[2*idx].active_material = self.mesh_material[idx]
            self.arrow_mesh[2*idx+1].active_material = self.mesh_material[idx]
        else:
            if idx == 0:
                self.mesh_material.append(bpy.data.materials.new('material'))
                self.mesh_material[0].diffuse_color = color_rgba[idx]
            self.arrow_mesh[2*idx].active_material = self.mesh_material[0]
            self.arrow_mesh[2*idx+1].active_material = self.mesh_material[0]

        # Set the diffusive color.
        if list_material:
            self.mesh_material[idx].diffuse_color = color_rgba[idx]
        else:
            self.mesh_material[0].diffuse_color = color_rgba[0]

        # Set the material roughness.
        if list_material:
            if isinstance(self.roughness, np.ndarray):
                self.mesh_material[idx].roughness = self.roughness[idx]
            else:
                self.mesh_material[idx].roughness = self.roughness
        elif idx == 0:
            self.mesh_material[0].roughness = self.roughness

        # Set the material emission.
        if not self.emission is None:
            if list_material:
                self.mesh_material[idx].use_nodes = True
                node_tree = self.mesh_material[idx].node_tree
                nodes = node_tree.nodes
                # Remove Diffusive BSDF node.
                nodes.remove(nodes[1])
                node_emission = nodes.new(type='ShaderNodeEmission')
                # Change the input of the ouput node to emission.
                node_tree.links.new(node_emission.outputs['Emission'],
                                    nodes[0].inputs['Surface'])
                # Adapt emission and color.
                node_emission.inputs['Color'].default_value = tuple(color_rgba[idx])
                if isinstance(self.emission, np.ndarray):
                    node_emission.inputs['Strength'].default_value = self.emission[idx]
                else:
                    node_emission.inputs['Strength'].default_value = self.emission
            else:
                self.mesh_material[0].use_nodes = True
                node_tree = self.mesh_material[0].node_tree
                nodes = node_tree.nodes
                # Remove Diffusive BSDF node.
                nodes.remove(nodes[1])
                node_emission = nodes.new(type='ShaderNodeEmission')
                # Change the input of the ouput node to emission.
                node_tree.links.new(node_emission.outputs['Emission'],
                                    nodes[0].inputs['Surface'])
                # Adapt emission and color.
                node_emission.inputs['Color'].default_value = color_rgba[idx]
                if isinstance(self.emission, np.ndarray):
                    node_emission.inputs['Strength'].default_value = self.emission
                else:
                    node_emission.inputs['Strength'].default_value = self.emission


    def time_handler(self, scene, depsgraph):
        """
        Function to be called whenever any Blender animation functions are used.
        Updates the plot according to the function specified.
        """

        if not self.time is None:
            self.plot()
        else:
            pass


    def update_globals(self):
        """
        Update the extrema and lights.
        """

        import blendaviz as blt

        if blt.house_keeping.x_min is None:
            blt.house_keeping.x_min = self.x.min()
        elif self.x.min() < blt.house_keeping.x_min:
            blt.house_keeping.x_min = self.x.min()
        if blt.house_keeping.x_max is None:
            blt.house_keeping.x_max = self.x.max()
        elif self.x.max() > blt.house_keeping.x_max:
            blt.house_keeping.x_max = self.x.max()

        if blt.house_keeping.y_min is None:
            blt.house_keeping.y_min = self.y.min()
        elif self.y.min() < blt.house_keeping.y_min:
            blt.house_keeping.y_min = self.y.min()
        if blt.house_keeping.y_max is None:
            blt.house_keeping.y_max = self.y.max()
        elif self.y.max() > blt.house_keeping.y_max:
            blt.house_keeping.y_max = self.y.max()

        if blt.house_keeping.z_min is None:
            blt.house_keeping.z_min = self.z.min()
        elif self.z.min() < blt.house_keeping.z_min:
            blt.house_keeping.z_min = self.z.min()
        if blt.house_keeping.z_max is None:
            blt.house_keeping.z_max = self.z.max()
        elif self.z.max() > blt.house_keeping.z_max:
            blt.house_keeping.z_max = self.z.max()

        if blt.house_keeping.box is None:
            blt.house_keeping.box = blt.bounding_box()
        else:
            blt.house_keeping.box.get_extrema()
            blt.house_keeping.box.plot()

        # Add some light.
        blt.adjust_lights()



def contour(phi, x, y, z, contours=1, psi=None,
            color=(0, 1, 0, 1), emission=None, roughness=1,
            vmin=None, vmax=None, color_map=None, time=None):
    """
    Plot contours to a given scalar field.

    Signature:

    contour(phi, x, y, z, contours=1, psi=None,
            color=(0, 1, 0, 1), emission=None, roughness=1,
            vmin=None, vmax=None, color_map=None, time=None):

    Signature:
    phi:  Scalar of shape (nx, ny, nz) or (nx, ny, nz, nt) for time dependent arrays..

    x, y, z:  x, y and z position of the data. These can be 1d arrays
        or 2d arrays of shape (nxyz, nt) for time dependent xyz.

    contours:  Number of contours to be plotted, or array of contour levels.

    psi:  Secondary scalar array of the same shape as phi, after which the
        contours are being textured.
        Use in conjunction with vmin and vmax.
        Recommended to use only one isosurface with psi.

    color:  rgba values of the form (r, g, b, a) with 0 <= r, g, b, a <= 1, or string,
        e.g. 'red' or character, e.g. 'r', or list of strings/character,
        or [n, 4] array with rgba values or array of the same shape as input array.

    emission:  Light emission by the contours. This overrides 'roughness'.

    roughness:  Texture roughness.

    vmin, vmax:  Minimum and maximum values for the colormap. If not specify, determine
        from the input arrays.

    color_map:  Color map for the values stored in the array 'c'.
        These are the same as in matplotlib.

    time:  Float array with the time information of the data.
        Has length nt.

    Returns
    -------
    3d Contour plot object.

    Examples
    ----
    >>> import numpy as np
    >>> import blendaviz as blt
    >>> x = np.linspace(-2, 2, 21)
    >>> y = np.linspace(-2, 2, 21)
    >>> z = np.linspace(-2, 2, 21)
    >>> xx, yy, zz = np.meshgrid(x, y, z)
    >>> phi = xx**2 + yy**2 + zz**2
    >>> iso = blt.contour(phi, xx, yy, zz, contours=[0.3, 0.6], color=np.array([(1, 0, 0, 1), (0, 1, 0, 0.5)]))
    """

    import inspect

    # Assign parameters to the arrow objects.
    contour_return = Contour3d()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(contour_return, argument, argument_dict[argument])
    contour_return.plot()
    return contour_return


class Contour3d(GenericPlot):
    """
    Contour class containing geometry, parameters and plotting function.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        import bpy
        import blendaviz as blt

        super().__init__()

        # Define the members that can be seen by the user.
        self.phi = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.psi = None
        self.contours = 1
        self.color = (0, 1, 0, 1)
        self.emission = None
        self.roughness = 1
        self.vmin = None
        self.vmax = None
        self.color_map = None
        self.mesh_data = None
        self.mesh_object = None
        self.mesh_material = None
        self.time = None
        self.time_index = 0
        self.deletable_object = None

        # Define the locally used time-independent data and parameters.
        self._phi = 0
        self._x = 0
        self._y = 0
        self._z = 0
        self._psi = None

        # Set the handler function for frame changes (time).
        bpy.app.handlers.frame_change_pre.append(self.time_handler)

        # Add the plot to the stack.
        blt.plot_stack.append(self)


    def plot(self):
        """
        Plot the contours.
        """

        import bpy
        import numpy as np
        from skimage import measure
        from blendaviz import colors

        # Check if there is any time array.
        if not self.time is None:
            if not isinstance(self.time, np.ndarray):
                print("Error: time is not a valid array.")
                return -1
            if self.time.ndim != 1:
                print("Error: time array must be 1d.")
                return -1
            # Determine the time index.
            self.time_index = np.argmin(abs(bpy.context.scene.frame_float - self.time))
        else:
            self.time = np.array([0])
            self.time_index = 0

        # Check the validity of the input arrays.
        if not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray) \
           or not isinstance(self.z, np.ndarray):
            print("Error: x OR y OR z array invalid.")
            return -1
        if not ((self.x.shape[0], self.y.shape[0], self.z.shape[0]) == self.phi.shape[:3]):
            print("Error: input array shapes invalid.")
            return -1
        if not self.psi is None:
            if not isinstance(self.psi, np.ndarray):
                print("Error: psi is not a numpy array.")
                return -1
            if not self.psi.shape[:3] == self.phi.shape[:3]:
                print("Error: psi and phi must of of the same shape.")

        # Point the local variables to the correct arrays.
        if self.x.ndim == 2:
            self._x = self.x[:, self.time_index]
        else:
            self._x = self.x
        if self.y.ndim == 2:
            self._y = self.y[:, self.time_index]
        else:
            self._y = self.y
        if self.z.ndim == 2:
            self._z = self.z[:, self.time_index]
        else:
            self._z = self.z
        if self.phi.ndim == 4:
            self._phi = self.phi[:, :, :, self.time_index]
        else:
            self._phi = self.phi
        if not self.psi is None:
            if self.psi.ndim == 4:
                self._psi = self.psi[:, :, :, self.time_index]
            else:
                self._psi = self.psi
        else:
            self._psi = None

        # Prepare the isosurface levels.
        if isinstance(self.contours, int):
            level_list = np.linspace(self._phi.min(),
                                     self._phi.max(),
                                     self.contours+2)[1:-1]
        elif isinstance(self.contours, list):
            level_list = np.array(self.contours)
        elif isinstance(self.contours, np.ndarray):
            level_list = self.contours.ravel()
        else:
            print("Error: countours invalid. \
                  Must be either integer or 1d array/list.")
            return -1

        # Prepare the material colors.
        color_rgba = colors.make_rgba_array(self.color, level_list.shape[0],
                                            self.color_map, self.vmin, self.vmax)

        # Determine the grid spacing.
        dx = np.partition(np.array(list(set(list(self._x.ravel())))), 1)[1] - \
             self._x.min()
        dy = np.partition(np.array(list(set(list(self._y.ravel())))), 1)[1] - \
             self._y.min()
        dz = np.partition(np.array(list(set(list(self._z.ravel())))), 1)[1] - \
             self._z.min()

        # Delete existing meshes.
        if not self.mesh_object is None:
            bpy.ops.object.select_all(action='DESELECT')
            if self.object_reference_valid(self.mesh_object):
                self.mesh_object.select_set(state=True)
                bpy.context.view_layer.objects.active = self.mesh_object
                bpy.ops.object.delete()
            self.mesh_object = None

        # Delete existing materials.
        if not self.mesh_material is None:
            for mesh_material in self.mesh_material:
                bpy.data.materials.remove(mesh_material)

        # Prepare the lists of mashes and materials.
        self.mesh_data = []
        self.mesh_object = []
        self.mesh_material = []

        for idx, level in enumerate(level_list):
            # Find the vertices and faces of the isosurfaces.
            vertices, faces, normals, values = measure.marching_cubes(self._phi,
                                                                      level,
                                                                      spacing=(dx, dy, dz))
            vertices[:, 0] += self._x.min()
            vertices[:, 1] += self._y.min()
            vertices[:, 2] += self._z.min()

            # Create mesh and object.
            self.mesh_data.append(bpy.data.meshes.new("DataMesh"))
            self.mesh_object.append(bpy.data.objects.new("ObjMesh", self.mesh_data[-1]))

            # Create mesh from the given data.
            self.mesh_data[-1].from_pydata(list(vertices), [], list(faces))
            self.mesh_data[-1].update(calc_edges=True)

            # Set the material/color.
            if self._psi is None:
                self.__set_material(idx, color_rgba, len(level_list))
            else:
                self.__color_vertices(idx, vertices)
            self.mesh_data[-1].materials.append(self.mesh_material[-1])

            # Link the mesh object with the scene.
            bpy.context.scene.collection.objects.link(self.mesh_object[-1])

        # Group the meshes together.
        for mesh in self.mesh_object[::-1]:
            mesh.select_set(state=True)
            bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.join()
        self.mesh_object = bpy.context.object
        self.mesh_object.select_set(state=False)

        # Make the grouped meshes the deletable object.
        self.deletable_object = self.mesh_object

        self.update_globals()

        return 0


    def __set_material(self, idx, color_rgba, n_levels):
        """
        Set the mesh material.

        Signature:

        __set_material(idx, color_rgba, n_levels):

        Parameters
        ----------
        idx:  Index of the material.

        color_rgba:  The rgba values of the colors to be used.

        n_levels:  Number of levels/isosurface.
        """

        import bpy
        import numpy as np

        # Deterimne if we need a list of materials, i.e. for every isosurface one.
        if any([isinstance(self.color, np.ndarray),
                isinstance(self.emission, np.ndarray),
                isinstance(self.roughness, np.ndarray)]):
            list_material = True
        else:
            list_material = False

        # Transform single values to arrays.
        if list_material:
            if color_rgba.shape[0] != n_levels:
                color_rgba = np.repeat(color_rgba, n_levels, axis=0)
            if not isinstance(self.roughness, np.ndarray):
                self.roughness = np.ones(n_levels)*self.roughness
            if not self.emission is None:
                if not isinstance(self.emission, np.ndarray):
                    self.emission = np.ones(n_levels)*self.emission

        # Set the material.
        if list_material:
            self.mesh_material.append(bpy.data.materials.new('material'))
        else:
            if idx == 0:
                self.mesh_material.append(bpy.data.materials.new('material'))
                self.mesh_material[0].diffuse_color = color_rgba[idx]

        # Set the diffusive color.
        if list_material:
            self.mesh_material[idx].diffuse_color = color_rgba[idx]
        else:
            self.mesh_material[0].diffuse_color = color_rgba[0]

        # Set the material roughness.
        if list_material:
            if isinstance(self.roughness, np.ndarray):
                self.mesh_material[idx].roughness = self.roughness[idx]
            else:
                self.mesh_material[idx].roughness = self.roughness
        elif idx == 0:
            self.mesh_material[0].roughness = self.roughness

        # Set the material emission.
        if not self.emission is None:
            if list_material:
                self.mesh_material[idx].use_nodes = True
                node_tree = self.mesh_material[idx].node_tree
                nodes = node_tree.nodes
                # Remove Diffusive BSDF node.
                nodes.remove(nodes[1])
                node_emission = nodes.new(type='ShaderNodeEmission')
                # Change the input of the ouput node to emission.
                node_tree.links.new(node_emission.outputs['Emission'],
                                    nodes[0].inputs['Surface'])
                # Adapt emission and color.
                node_emission.inputs['Color'].default_value = tuple(color_rgba[idx])
                if isinstance(self.emission, np.ndarray):
                    node_emission.inputs['Strength'].default_value = self.emission[idx]
                else:
                    node_emission.inputs['Strength'].default_value = self.emission
            else:
                self.mesh_material[0].use_nodes = True
                node_tree = self.mesh_material[0].node_tree
                nodes = node_tree.nodes
                # Remove Diffusive BSDF node.
                nodes.remove(nodes[1])
                node_emission = nodes.new(type='ShaderNodeEmission')
                # Change the input of the ouput node to emission.
                node_tree.links.new(node_emission.outputs['Emission'],
                                    nodes[0].inputs['Surface'])
                # Adapt emission and color.
                node_emission.inputs['Color'].default_value = color_rgba[idx]
                if isinstance(self.emission, np.ndarray):
                    node_emission.inputs['Strength'].default_value = self.emission
                else:
                    node_emission.inputs['Strength'].default_value = self.emission


    def __color_vertices(self, idx, vertices):
        """
        Set the mesh texture.

        Signature:

        __color_vertices(idx, vertices):

        Parameters
        ----------
        idx:  Index of the material.

        vertices:  Vertices of the isosurfaces.
        """

        import bpy
        import numpy as np
        from blendaviz import colors

        # Find interpolated values for Psi on the vertex location.
        psi_vertices = np.zeros(vertices.shape[0])
        for vertex_idx in range(vertices.shape[0]):
            vertex = vertices[vertex_idx]
            # Find the xyz indices for the interpolation.
            ix1 = sum(self._x <= vertex[0]) - 1
            iy1 = sum(self._y <= vertex[1]) - 1
            iz1 = sum(self._z <= vertex[2]) - 1
            ix2 = ix1 + 1
            iy2 = iy1 + 1
            iz2 = iz1 + 1
            if ix2 >= self._phi.shape[0]:
                ix2 = self._phi.shape[0]
            if iy2 >= self._phi.shape[1]:
                iy2 = self._phi.shape[1]
            if iz2 >= self._phi.shape[2]:
                iz2 = self._phi.shape[2]
            # Perform a trilinear interpolation.
            psi_vertices[vertex_idx] = np.mean(self._psi[ix1:ix2+1,
                                                         iy1:iy2+1,
                                                         iz1:iz2+1])

        # Generate the colors for the vertices.
        color_rgba = colors.make_rgba_array(psi_vertices, vertices.shape[0],
                                            self.color_map, self.vmin, self.vmax)

        # Create a vertex color layer for the mesh.
        vcol_layer = self.mesh_object[idx].data.vertex_colors.new()

        # Add a new material.
        self.mesh_material.append(bpy.data.materials.new('material'))
        self.mesh_material[-1].use_nodes = True
        node_tree = self.mesh_material[-1].node_tree
        nodes = node_tree.nodes
        nodes.remove(nodes[1])
        node_diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
        node_tree.links.new(node_diffuse.outputs['BSDF'], nodes[0].inputs['Surface'])
        node_vertex_shader = nodes.new(type='ShaderNodeVertexColor')
        node_tree.links.new(node_vertex_shader.outputs['Color'], node_diffuse.inputs['Color'])
        node_vertex_shader.layer_name = 'Col'

        # Change the color of the vertices.
        for poly in self.mesh_object[idx].data.polygons:
            for loop_index in poly.loop_indices:
                loop_vert_index = self.mesh_object[idx].data.loops[loop_index].vertex_index
                vcol_layer.data[loop_index].color = color_rgba[loop_vert_index]


    def time_handler(self, scene, depsgraph):
        """
        Function to be called whenever any Blender animation functions are used.
        Updates the plot according to the function specified.
        """

        if not self.time is None:
            self.plot()
        else:
            pass


    def update_globals(self):
        """
        Update the extrema and lights.
        """

        import blendaviz as blt

        if blt.house_keeping.x_min is None:
            blt.house_keeping.x_min = self.x.min()
        elif self.x.min() < blt.house_keeping.x_min:
            blt.house_keeping.x_min = self.x.min()
        if blt.house_keeping.x_max is None:
            blt.house_keeping.x_max = self.x.max()
        elif self.x.max() > blt.house_keeping.x_max:
            blt.house_keeping.x_max = self.x.max()

        if blt.house_keeping.y_min is None:
            blt.house_keeping.y_min = self.y.min()
        elif self.y.min() < blt.house_keeping.y_min:
            blt.house_keeping.y_min = self.y.min()
        if blt.house_keeping.y_max is None:
            blt.house_keeping.y_max = self.y.max()
        elif self.y.max() > blt.house_keeping.y_max:
            blt.house_keeping.y_max = self.y.max()

        if blt.house_keeping.z_min is None:
            blt.house_keeping.z_min = self.z.min()
        elif self.z.min() < blt.house_keeping.z_min:
            blt.house_keeping.z_min = self.z.min()
        if blt.house_keeping.z_max is None:
            blt.house_keeping.z_max = self.z.max()
        elif self.z.max() > blt.house_keeping.z_max:
            blt.house_keeping.z_max = self.z.max()

        if blt.house_keeping.box is None:
            blt.house_keeping.box = blt.bounding_box()
        else:
            blt.house_keeping.box.get_extrema()
            blt.house_keeping.box.plot()

        # Add some light.
        blt.adjust_lights()


#def vol(phi, x, y, z, emission=None, color_map=None):
    #"""
    #Plot a 3d volume rendering of a scalar field.

    #Signature:

    #vol(phi, x, y, z, alpha=None, color_map=None)

    #Parameters
    #----------
    #phi:  Scalar field in 3d of shape (nx, ny, nz).

    #x, y, z:  1d arrays for the coordinates.

    #emission:  1d or 2d array of floats containing the emission values for phi.
        #If specified as 2d array fo shape (2, n_emission) the emission[0, :] are the values
        #phi which must be monotonously increasing and emission[1, ;] are their corresponding
        #emission values.

    #color_map:  Color map for the values stored in the array 'c'.
        #These are the same as in matplotlib.

    #Returns
    #-------
    #Volume plot object.
    #"""

    #import inspect

    ## Assign parameters to the Volume objects.
    #volume_return = Volume()
    #argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    #for argument in argument_dict:
        #setattr(volume_return, argument, argument_dict[argument])
    #volume_return.plot()
    #return volume_return


#class Volume(GenericPlot):
    #"""
    #Volume class including the data, 3d texture and parameters.
    #"""

    #def __init__(self):
        #"""
        #Fill members with default values.
        #"""

        #self.phi = 0
        #self.x = 0
        #self.y = 0
        #self.z = 0
        #self.emission = None
        #self.color_map = None
        #self.mesh_object = None
        #self.mesh_material = None


    #def plot(self):
        #"""
        #Plot the 3d texture.
        #"""

        #import bpy
        #import numpy as np
        #import matplotlib.cm as cm

        ## Check the validity of the input arrays.
        #if (not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray)
                #or not isinstance(self.z, np.ndarray)):
            #print("Error: x OR y OR z array invalid.")
            #return -1
        #if not isinstance(self.phi, np.ndarray):
            #print("Error: phi must be numpy array.")
            #return -1
        #if self.emission:
            #if not isinstance(self.emission, np.ndarray):
                #print("Error: emission must be numpy array.")
                #return -1

        ## Using volumetric textures or voxels?

        ## Delete existing meshes.
        #if not self.mesh_object is None:
            #bpy.ops.object.select_all(action='DESELECT')
            #self.mesh_object.select = True
            #bpy.ops.object.delete()
            #self.mesh_object = None

        ## Delete existing materials.
        #if not self.mesh_material is None:
            #bpy.data.materials.remove(self.mesh_material)

        ## Create cuboid.
        #bpy.ops.mesh.primitive_cube_add()
        #self.mesh_object = bpy.context.object

        ## Define the RGB value for each voxel.
        #phi_max = np.max(self.phi)
        #phi_min = np.min(self.phi)
        #if self.color_map is None:
            #self.color_map = cm.viridis
##        pixels[0::3] = self.color_map((self.phi.flatten() - phi_min)/(phi_max - phi_min))[:, 0]
##        pixels[1::3] = self.color_map((self.phi.flatten() - phi_min)/(phi_max - phi_min))[:, 1]
##        pixels[2::3] = self.color_map((self.phi.flatten() - phi_min)/(phi_max - phi_min))[:, 2]

        ## Define the emission for each voxel.
        #emission = np.zeros_like(self.phi)
        #if self.emission.ndim == 1:
            #phi_emission = np.zeros([2, self.emission.size])
            #phi_emission[0, :] = np.linspace(self.phi.min(), self.size.max(), self.emission.size)
            #phi_emission[1, :] = self.emission
        #else:
            #phi_emission = self.emission

        #for emission_idx in self.emission.shape[-1]-1:
            #mask = self.phi >= phi_emission[0, emission_idx] and self.phi < phi_emission[0, emission_idx+1]
            #weight_left = self.phi[mask] - phi_emission[0, emission_idx]
            #weight_right = -self.phi[mask] + phi_emission[0, emission_idx+1]
            #emission[mask] = (weight_left*phi_emission[1, emission_idx] + \
                              #weight_right*phi_emission[1, emission_idx+1]) \
                             #/(weight_left + weight_right)
        #del(emission)

        ## Assign a material to the cuboid.
        #self.mesh_material = bpy.data.materials.new('MaterialMesh')
        #self.mesh_material.use_nodes = True

        ## Add the RGB and emission values to the material.
        #nodes = node_tree.nodes
        ## Remove diffusive BSDF node.
        #nodes.remove(nodes[1])
        ## Add the emission shader.
        #node_emission = nodes.new(type='ShaderNodeEmission')
        ## Link the emission output to the material output.
        #node_tree.links.new(node_emission.outputs['Emission'],
                            #nodes[0].inputs['Volume'])
        ## Add the RGB source node.

        ## Link the RGB output to the emission shader  color input.

        ## Add the emission source node.

        ## Link the emission output to the emission shader strength input.


        #return 0
