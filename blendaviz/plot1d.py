# plot1d.py
"""
Contains routines to one-dimensional plots.
"""

from typing import Optional, Union, Tuple, List, Any
import numpy as np
import numpy.typing as npt

from blendaviz.generic import GenericPlot


def plot(
    x: Union[npt.NDArray[np.floating], List[float]],
    y: Union[npt.NDArray[np.floating], List[float]],
    z: Union[npt.NDArray[np.floating], List[float]],
    radius: Union[float, npt.NDArray[np.floating]] = 0.1,
    resolution: int = 8,
    color: Union[Tuple, str, List, npt.NDArray[np.floating]] = (0, 1, 0, 1),
    emission: Optional[Union[float, npt.NDArray[np.floating]]] = None,
    roughness: Union[float, npt.NDArray[np.floating]] = 1,
    rotation_x: Union[float, npt.NDArray[np.floating]] = 0,
    rotation_y: Union[float, npt.NDArray[np.floating]] = 0,
    rotation_z: Union[float, npt.NDArray[np.floating]] = 0,
    marker: Optional[Union[str, Any]] = None,
    time: Optional[npt.NDArray[np.floating]] = None
) -> 'PathLine':
    """
    Line plot in 3 dimensions as a line, tube or shapes.

    Signature:

    plot(x, y, z, radius=0.1, resolution=8, color=(0, 1, 0, 1),
         =None, roughness=1, rotation_x=0, rotation_y=0, rotation_z=0,
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

    # Capture arguments before creating any local variables.
    arguments = dict(locals())

    # Assign parameters to the PathLine objects.
    path_line_return = PathLine()
    for key, value in arguments.items():
        setattr(path_line_return, key, value)

    # Plot the data.
    path_line_return.plot()
    return path_line_return



class PathLine(GenericPlot):
    """
    Path line class including the vertices, parameters and plotting function.
    """

    # Type hints for main attributes
    x: Union[float, npt.NDArray[np.floating]]
    y: Union[float, npt.NDArray[np.floating]]
    z: Union[float, npt.NDArray[np.floating]]
    radius: Union[float, npt.NDArray[np.floating]]
    resolution: int
    color: Union[Tuple, str, List, npt.NDArray[np.floating]]
    emission: Optional[Union[float, npt.NDArray[np.floating]]]
    roughness: Union[float, npt.NDArray[np.floating]]
    rotation_x: Union[float, npt.NDArray[np.floating]]
    rotation_y: Union[float, npt.NDArray[np.floating]]
    rotation_z: Union[float, npt.NDArray[np.floating]]
    marker: Optional[Union[str, Any]]
    time_index: int
    curve_data: Optional[Any]  # bpy.types.Curve
    curve_object: Optional[Any]  # bpy.types.Object
    marker_mesh: Optional[List[Any]]  # List of bpy.types.Mesh
    mesh_material: Optional[Any]  # bpy.types.Material
    mesh_texture: Optional[Any]  # bpy.types.ShaderNodeTexImage
    poly_line: Optional[Any]  # bpy.types.Spline
    deletable_object: Optional[Any]  # bpy.types.Object

    def __init__(self) -> None:
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
        self.radius = 0.1
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.rotation = (0, 0, 0)
        self.color = (0, 1, 0, 1)
        self.roughness = 1.0
        self.emission = None
        self.marker = None
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

        # Add the plot to the stack.
        blt.plot_stack.append(self)


    def plot(self) -> None:
        """
        Plot a as a line, tube or shapes.
        """

        import bpy
        import numpy as np
        from blendaviz import colors, markers, materials

        # Check if there is any time array.
        if self.time is not None:
            if not isinstance(self.time, np.ndarray):
                raise TypeError("time must be a numpy array")
            if self.time.ndim != 1:
                raise ValueError("time array must be 1-dimensional")
            # Determine the time index.
            self.time_index = np.argmin(abs(bpy.context.scene.frame_float - self.time))
        else:
            self.time = np.array([0])
            self.time_index = 0

        # Convert lists to arrays.
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)

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
        if self.curve_data is not None:
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
                self.mesh_material.diffuse_color = color_rgba
            self.mesh_material.roughness = self.roughness
            self.curve_object.active_material = self.mesh_material

            # Set the emission.
            if self.emission is not None:
                self.mesh_material.use_nodes = True
                node_tree = self.mesh_material.node_tree
                nodes = node_tree.nodes

                # Find the material output node
                output_node = None
                for node in nodes:
                    if node.type == 'OUTPUT_MATERIAL':
                        output_node = node
                        break

                # Remove any Diffusive BSDF node.
                for node in list(nodes):
                    if node != output_node:
                        nodes.remove(node)

                # Create the emission node.
                node_emission = nodes.new(type='ShaderNodeEmission')

                # Change the input of the output node to emission.
                node_tree.links.new(node_emission.outputs['Emission'],
                                    output_node.inputs['Surface'])

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
        if self.marker is not None:
            self.marker_mesh = []

            # Use marker factory for standard marker types
            if isinstance(self.marker, str) and self.marker in markers.MARKER_FACTORIES:
                self.marker_mesh = markers.create_markers(
                    self.marker,
                    self._x, self._y, self._z,
                    self._radius,
                    self._rotation_x, self._rotation_y, self._rotation_z
                )

                # Apply smooth shading for spheres
                if self.marker in ['ico_sphere', 'uv_sphere']:
                    for marker_obj in self.marker_mesh:
                        bpy.context.view_layer.objects.active = marker_obj
                        bpy.ops.object.shade_smooth()

            # Handle custom mesh objects
            elif isinstance(self.marker, bpy.types.Object):
                if self.marker.type == 'MESH':
                    bpy.context.object.select_set(False)
                    self.marker.select_set(True)
                    for idx in range(self._x.shape[0]):
                        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'})
                        bpy.context.object.location = (self._x[idx], self._y[idx], self._z[idx])
                        bpy.context.object.rotation_euler = (self._rotation_x[idx], self._rotation_y[idx], self._rotation_z[idx])
                        self.marker.select_set(False)
                        self.marker_mesh.append(bpy.context.object)

            # Set the material and color.
            if self.marker is not None:
                color_is_array = False
                if isinstance(color_rgba, np.ndarray):
                    if color_rgba.ndim == 2:
                        color_is_array = True

                if any([color_is_array,
                        isinstance(self.roughness, np.ndarray),
                        isinstance(self.emission, np.ndarray)]):
                    self.mesh_material = []

                    for idx in range(self._x.shape[0]):
                        # Get color for this marker
                        marker_color = tuple(color_rgba[idx]) if color_is_array else color_rgba

                        # Get roughness for this marker
                        marker_roughness = self.roughness[idx] if isinstance(self.roughness, np.ndarray) else self.roughness

                        # Get emission for this marker
                        marker_emission = self.emission[idx] if isinstance(self.emission, np.ndarray) else None

                        # Create material using helper
                        material = materials.create_material_with_color(
                            f'MarkerMaterial{idx}',
                            marker_color,
                            marker_emission,
                            marker_roughness
                        )
                        self.mesh_material.append(material)
                        self.marker_mesh[idx].active_material = material
                else:
                    # Create single material for all markers using helper
                    self.mesh_material = materials.create_material_with_color(
                        'MarkerMaterial',
                        color_rgba,
                        self.emission,
                        self.roughness
                    )

                    # Also set diffuse_color for compatibility
                    self.mesh_material.diffuse_color = color_rgba

                    # Apply material to all markers
                    for idx, mesh in enumerate(self.marker_mesh):
                        mesh.active_material = self.mesh_material

            # Group the meshes together.
            if self.marker is not None:
                for mesh in self.marker_mesh[::-1]:
                    mesh.select_set(state=True)
                # Only join if there are multiple objects selected.
                if len(bpy.context.selected_objects) > 1:
                    bpy.ops.object.join()
                self.marker_mesh = bpy.context.object
                self.marker_mesh.select_set(state=False)
                # Make the grouped meshes the deletable object.
                self.deletable_object = self.marker_mesh

        self.update_globals()

        return 0


    def __delete_meshes__(self) -> None:
        """
        Delete all existing meshes that are part of this plot.
        """

        import bpy

        if self.marker_mesh is not None:
            bpy.ops.object.select_all(action='DESELECT')
            if self.object_reference_valid(self.marker_mesh):
                self.marker_mesh.select_set(True)
                bpy.context.view_layer.objects.active = self.marker_mesh
                bpy.ops.object.delete()
            self.marker_mesh = None


    def __delete_materials__(self) -> None:
        """
        Delete all existing meshes that are part of this plot.
        """

        import bpy

        if self.mesh_material is not None:
            if isinstance(self.mesh_material, list):
                for mesh_material in self.mesh_material:
                    if self.object_reference_valid(mesh_material):
                        bpy.data.materials.remove(mesh_material)
            else:
                if self.object_reference_valid(self.mesh_material):
                    bpy.data.materials.remove(self.mesh_material)
            self.mesh_material = None


    def update_globals(self) -> None:
        """
        Update the extrema, camera and lights.
        """

        import blendaviz as blt

        self.update_extrema_x()
        self.update_extrema_y()
        self.update_extrema_z()

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
