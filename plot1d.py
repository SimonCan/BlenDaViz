# plot1d.py
"""
Contains routines to one-dimensional plots.
"""


from blendaviz.generic import GenericPlot


def plot(x, y, z, radius=0.1, resolution=8, color=(0, 1, 0, 1),
         emission=None, roughness=1, rotation_x=0, rotation_y=0, rotation_z=0,
         marker=None, time=None):
    """
    Line plot in 3 dimensions as a line, tube or shapes.

    Signature:

    plot(x, y, z, radius=0.1, resolution=8, color=(0, 1, 0, 1),
         emission=None, roughness=1, rotation_x=0, rotation_y=0, rotation_z=0,
         marker=None, time=None)

    Parameters
    ----------
    x, y, z:  x, y and z coordinates of the points to be plotted.
        These are 1d arrays of the same length n
        or 2d time dependent arrays of shape (n, nt).

    radius:  Radius of the plotted tube, i.e. line width, or size of the markers.
        Positive real number for point and time independenet radius
        or 1d array of length n for point dependenet radius
        or 2d array of length (n, nt) for point and time dependenet radius
        or 2d array of length (1, nt) for time dependenet radius.

    rotation_[xyz]: Rotation angle around the xyz axis.
        Real number for point and time independent radius
        or array of length n for point dependent radius
        or 2d array of length (n, nt) for point and time dependent radius
        or 2d array of length (1, nt) for time dependent radius.

    resolution:  Azimuthal resolution of the tubes in vertices.
        Positive integer > 2.

    color:  rgba values of the form (r, g, b, a) with 0 <= r, g, b, a <= 1, or string,
        e.g. 'red', or character, e.g. 'r', or n-array of strings/character,
        or [n, 4] array with rgba values.

    emission:  Light emission of the line or markers.
        Real number for a line plot and array for markers.

    roughness:  Texture roughness.

    marker:  Marker to be used for the plot.
        String with standard Blender 3d shapes: 'cube', 'uv_sphere', 'ico_sphere',
        'cylinder', 'cone', 'torus', 'monkey'.
        Custom shape or blender object.
        1d array of length n of one of the above.

    time:  Float array with the time information of the data.
        Has length nt.

    Returns
    -------
    1d PathLine object.

    Examples
    --------
    >>> import numpy as np
    >>> import blendaviz as blt
    >>> z = np.linspace(0, 6*np.pi, 30)
    >>> x = 3*np.cos(z)
    >>> y = 3*np.sin(z)
    >>> pl = blt.plot(x, y, z, marker='cube', radius=0.5, rotation_x=z, rotation_y=np.zeros_like(x), rotation_z=np.zeros_like(x))
    >>> pl.colors = np.random.random([x.shape[0], 3])
    >>> pl.z = np.linspace(0, 6, 30)
    >>> pl.plot()
    """

    import inspect

    # Assign parameters to the PathLine objects.
    path_line_return = PathLine()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(path_line_return, argument, argument_dict[argument])

    # Plot the data.
    path_line_return.plot()
    return path_line_return



class PathLine(GenericPlot):
    """
    Path line class including the vertices, parameters and plotting function.
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
        self.z = 0
        self.radius = 0.1
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.rotation = (0, 0, 0)
        self.color = (0, 1, 0, 1)
        self.roughness = 1.0
        self.emission = None
        self.marker = None
        self.time = None
        self.time_index = 0
        self.curve_data = None
        self.curve_object = None
        self.marker_mesh = None
        self.mesh_material = None
        self.mesh_texture = None
        self.poly_line = None
        self.deletable_object = None

        # Define the locally used time-independent data and parameters.
        self._x = 0
        self._y = 0
        self._z = 0
        self._radius = 0.1
        self._rotation_x = 0
        self._rotation_y = 0
        self._rotation_z = 0

        # Set the handler function for frame changes (time).
        bpy.app.handlers.frame_change_pre.append(self.time_handler)

        # Add the plot to the stack.
        blt.__stack__.append(self)


    def plot(self):
        """
        Plot a as a line, tube or shapes.
        """

        import bpy
        import numpy as np
        from blendaviz import colors

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

        # Point the local variables to the correct time index.
        arrays_with_time_list = ['x', 'y', 'z', 'radius', 'rotation_x', 'rotation_y', 'rotation_z']
        for array_with_time in arrays_with_time_list:
            array_value = getattr(self, array_with_time)
            if not isinstance(array_value, np.ndarray):
                setattr(self, '_' + array_with_time, array_value*np.ones(self.x.shape[0]))
            elif array_value.ndim == 1:
                setattr(self, '_' + array_with_time, array_value)
            else:
                setattr(self, '_' + array_with_time, array_value[:, self.time_index])

        # Delete existing curve.
        if not self.curve_data is None:
            bpy.data.curves.remove(self.curve_data)
            self.curve_data = None

        # Delete existing meshes.
        self.__delete_meshes__()

        # Delete existing materials.
        self.__delete_materials__()

        # Create the bezier curve.
        if self.marker is None:
            color_is_array = False
            # Transform color string into rgba.
            if isinstance(self.color, np.ndarray):
                color_rgba = colors.make_rgba_array(self.color, self._x.shape[0])
                color_is_array = True
            else:
                color_rgba = colors.make_rgba_array(self.color, 1)

            self.curve_data = bpy.data.curves.new('DataCurve', type='CURVE')
            self.curve_data.dimensions = '3D'
            self.curve_object = bpy.data.objects.new('ObjCurve', self.curve_data)

            # Set the origin to the last point.
            self.curve_object.location = tuple((self._x[-1], self._y[-1], self._z[-1]))

            # Add the rest of the curve.
            self.poly_line = self.curve_data.splines.new('POLY')
            self.poly_line.points.add(self._x.shape[0])
            for param in range(self._x.shape[0]):
                self.poly_line.points[param].co = (self._x[param] - self._x[-1],
                                                   self._y[param] - self._y[-1],
                                                   self._z[param] - self._z[-1],
                                                   0)

            # Add 3d structure.
            self.curve_data.splines.data.bevel_depth = self._radius[0]
            self.curve_data.splines.data.bevel_resolution = self.resolution
            self.curve_data.splines.data.fill_mode = 'FULL'

            # Set the material/color.
            self.mesh_material = bpy.data.materials.new('material')
            if color_is_array:
                # Assign the texture to the material.
                self.mesh_material.use_nodes = True
                self.mesh_texture = self.mesh_material.node_tree.nodes.new('ShaderNodeTexImage')
                self.mesh_texture.extension = 'EXTEND'
                # Prepare the image texture.
                mesh_image = bpy.data.images.new('ImageMesh', self.color.shape[0], 1)
                pixels = np.array(mesh_image.pixels)
                # Assign the RGBa values to the pixels.
                pixels[0::4] = color_rgba[:, 0]
                pixels[1::4] = color_rgba[:, 1]
                pixels[2::4] = color_rgba[:, 2]
                pixels[3::4] = 1
                mesh_image.pixels[:] = np.swapaxes(pixels.reshape([self.color.shape[0],
                                                                   1, 4]), 0, 1).flatten()[:]
                self.mesh_texture.image = mesh_image
                links = self.mesh_material.node_tree.links
                links.new(self.mesh_texture.outputs[0],
                          self.mesh_material.node_tree.nodes.get("Principled BSDF").inputs[0])
            else:
                self.mesh_material.diffuse_color = color_rgba[0]
            self.mesh_material.roughness = self.roughness
            self.curve_object.active_material = self.mesh_material

            # Set the emission.
            if not self.emission is None:
                self.mesh_material.use_nodes = True
                node_tree = self.mesh_material.node_tree
                nodes = node_tree.nodes
                # Remove and Diffusive BSDF node.
                nodes.remove(nodes[1])
                node_emission = nodes.new(type='ShaderNodeEmission')
                # Change the input of the ouput node to emission.
                node_tree.links.new(node_emission.outputs['Emission'],
                                    nodes[0].inputs['Surface'])
                # Adapt emission and color.
                node_emission.inputs['Color'].default_value = color_rgba
                node_emission.inputs['Strength'].default_value = self.emission

            # Link the curve object with the scene.
            bpy.context.scene.collection.objects.link(self.curve_object)

            # Make this curve the object to be deleted.
            self.deletable_object = self.curve_object

        # Transform color string into rgb.
        color_rgba = colors.make_rgba_array(self.color, self._x.shape[0])

        # Plot the markers.
        if not self.marker is None:
            self.marker_mesh = []
        if self.marker == 'cone':
            for idx in range(self._x.shape[0]):
                bpy.ops.mesh.primitive_cone_add(location=(self._x[idx], self._y[idx], self._z[idx]),
                                                radius1=self._radius[idx],
                                                depth=2*self._radius[idx],
                                                rotation=(self._rotation_x[idx],
                                                          self._rotation_y[idx],
                                                          self._rotation_z[idx]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'cube':
            for idx in range(self._x.shape[0]):
                bpy.ops.mesh.primitive_cube_add(location=(self._x[idx], self._y[idx], self._z[idx]),
                                                size=self._radius[idx],
                                                rotation=(self._rotation_x[idx],
                                                          self._rotation_y[idx],
                                                          self._rotation_z[idx]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'cylinder':
            for idx in range(self._x.shape[0]):
                bpy.ops.mesh.primitive_cylinder_add(location=(self._x[idx], self._y[idx], self._z[idx]),
                                                    radius=self._radius[idx],
                                                    depth=2*self._radius[idx],
                                                    rotation=(self._rotation_x[idx],
                                                              self._rotation_y[idx],
                                                              self._rotation_z[idx]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'ico_sphere':
            for idx in range(self._x.shape[0]):
                bpy.ops.mesh.primitive_ico_sphere_add(location=(self._x[idx], self._y[idx], self._z[idx]),
                                                      radius=self._radius[idx],
                                                      rotation=(self._rotation_x[idx],
                                                                self._rotation_y[idx],
                                                                self._rotation_z[idx]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'monkey':
            for idx in range(self._x):
                bpy.ops.mesh.primitive_monkey_add(location=(self._x[idx], self._y[idx], self._z[idx]),
                                                  size=self._radius[idx],
                                                  rotation=(self._rotation_x[idx],
                                                            self._rotation_y[idx],
                                                            self._rotation_z[idx]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'torus':
            for idx in range(self._x.shape[0]):
                bpy.ops.mesh.primitive_torus_add(location=(self._x[idx], self._y[idx], self._z[idx]),
                                                 major_radius=self._radius[idx],
                                                 minor_radius=0.25*self._radius[idx],
                                                 abso_major_rad=1.25*self._radius[idx],
                                                 abso_minor_rad=0.75*self._radius[idx],
                                                 rotation=(self._rotation_x[idx],
                                                           self._rotation_y[idx],
                                                           self._rotation_z[idx]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'uv_sphere':
            for idx in range(self._x.shape[0]):
                bpy.ops.mesh.primitive_uv_sphere_add(location=(self._x[idx], self._y[idx], self._z[idx]),
                                                     radius=self._radius[idx],
                                                     rotation=(self._rotation_x[idx],
                                                               self._rotation_y[idx],
                                                               self._rotation_z[idx]))
                self.marker_mesh.append(bpy.context.object)
        if isinstance(self.marker, bpy.types.Object):
            if self.marker.type == 'MESH':
                bpy.context.object.select = False
                self.marker.select = True
                for idx in range(self._x.shape[0]):
                    bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'})
                    bpy.context.object.location = (self._x[idx], self._y[idx], self._z[idx])
                    bpy.context.object.rotation_euler = (self._rotation_x[idx], self._rotation_y[idx], self._rotation_z[idx])
                    self.marker.select = False
                    self.marker_mesh.append(bpy.context.object)

        # Set the material and color.
        if not self.marker is None:
            color_is_array = False
            if isinstance(color_rgba, np.ndarray):
                if color_rgba.ndim == 2:
                    color_is_array = True

            if any([color_is_array,
                    isinstance(self.roughness, np.ndarray),
                    isinstance(self.emission, np.ndarray)]):
                self.mesh_material = []

                for idx in range(self._x.shape[0]):
                    self.mesh_material.append(bpy.data.materials.new('material'))

                    if color_is_array:
                        self.mesh_material[idx].diffuse_color = tuple(color_rgba[idx])
                    else:
                        self.mesh_material[idx].diffuse_color = color_rgba

                    if isinstance(self.roughness, np.ndarray):
                        self.mesh_material[idx].roughness = self.roughness[idx]
                    else:
                        self.mesh_material[idx].roughness = self.roughness

                    if isinstance(self.emission, np.ndarray):
                        self.mesh_material[idx].use_nodes = True
                        node_tree = self.mesh_material[idx].node_tree
                        nodes = node_tree.nodes
                        # Remove and Diffusive BSDF node.
                        nodes.remove(nodes[1])
                        node_emission = nodes.new(type='ShaderNodeEmission')
                        # Change the input of the ouput node to emission.
                        node_tree.links.new(node_emission.outputs['Emission'],
                                            nodes[0].inputs['Surface'])
                        # Adapt emission and color.
                        if color_is_array:
                            node_emission.inputs['Color'].default_value = tuple(color_rgba[idx])
                        else:
                            node_emission.inputs['Color'].default_value = color_rgba
                        node_emission.inputs['Strength'].default_value = self.emission[idx]

                    self.marker_mesh[idx].active_material = self.mesh_material[idx]
            else:
                self.mesh_material = bpy.data.materials.new('material')
                self.mesh_material.diffuse_color = color_rgba
                self.mesh_material.roughness = self.roughness

                if not self.emission is None:
                    self.mesh_material.use_nodes = True
                    node_tree = self.mesh_material.node_tree
                    nodes = node_tree.nodes
                    # Remove and Diffusive BSDF node.
                    nodes.remove(nodes[1])
                    node_emission = nodes.new(type='ShaderNodeEmission')
                    # Change the input of the ouput node to emission.
                    node_tree.links.new(node_emission.outputs['Emission'],
                                        nodes[0].inputs['Surface'])
                    # Adapt emission and color.
                    node_emission.inputs['Color'].default_value = color_rgba
                    node_emission.inputs['Strength'].default_value = self.emission

                for idx, mesh in enumerate(self.marker_mesh):
                    mesh.active_material = self.mesh_material

        # Group the meshes together.
        if not self.marker is None:
            for mesh in self.marker_mesh[::-1]:
                mesh.select_set(state=True)
            bpy.ops.object.join()
            self.marker_mesh = bpy.context.object
            self.marker_mesh.select_set(state=False)
            # Make the grouped meshes the deletable object.
            self.deletable_object = self.marker_mesh

        self.update_globals()


    def __delete_meshes__(self):
        """
        Delete all existing meshes that are part of this plot.
        """

        import bpy

        if not self.marker_mesh is None:
            bpy.ops.object.select_all(action='DESELECT')
            if self.object_reference_valid(self.marker_mesh):
                if isinstance(self.marker_mesh, list):
                    for marker_mesh in self.marker_mesh:
                        if self.object_reference_valid(marker_mesh):
                            marker_mesh.select_set(True)
                            bpy.context.view_layer.objects.active = marker_mesh
                            bpy.ops.object.delete()
                else:
                    self.marker_mesh.select_set(True)
                    print(type(self.marker_mesh))
                    bpy.context.view_layer.objects.active = self.marker_mesh
                    bpy.ops.object.delete()
            self.marker_mesh = None


    def __delete_materials__(self):
        """
        Delete all existing meshes that are part of this plot.
        """

        import bpy

        if not self.mesh_material is None:
            if isinstance(self.mesh_material, list):
                for mesh_material in self.mesh_material:
                    if self.object_reference_valid(mesh_material):
                        bpy.data.materials.remove(mesh_material)
            else:
                if self.object_reference_valid(self.mesh_material):
                    bpy.data.materials.remove(self.mesh_material)
            self.mesh_material = None


    def update_globals(self):
        """
        Update the extrema, camera and lights.
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

        # Add or update bounding box.
        if blt.house_keeping.box is None:
            blt.house_keeping.box = blt.bounding_box()
        else:
            blt.house_keeping.box.get_extrema()
            blt.house_keeping.box.plot()

        # Add some light.
        blt.adjust_lights()

        # Add a camera.
        blt.adjust_camera()