# streamlines3d.py
"""
Contains routines to generate and plot streamlines.
"""

from typing import Optional, Union, Tuple, List, Callable, Any
import numpy as np
import numpy.typing as npt

from blendaviz.generic import GenericPlot


def streamlines_function(
    field_function: Callable,
    n_seeds: int = 100,
    seeds: Optional[npt.NDArray[np.floating]] = None,
    seed_center: Optional[Union[Tuple[float, float, float], npt.NDArray[np.floating]]] = None,
    seed_radius: float = 1,
    method: str = 'DOP853',
    atol: float = 1e-4,
    rtol: float = 1e-4,
    metric: Optional[Callable] = None,
    integration_time: float = 1,
    integration_steps: int = 10,
    integration_direction: str = 'both',
    color: Union[Tuple, str, List, npt.NDArray[np.floating]] = (0, 1, 0, 1),
    color_scalar: Optional[Union[str, Callable]] = None,
    emission: Optional[Union[float, npt.NDArray[np.floating]]] = None,
    roughness: float = 1,
    radius: float = 0.1,
    resolution: int = 8,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    color_map: Optional[Any] = None,
    n_proc: int = 1
) -> 'Streamlines3d':
    """
    Plot streamlines of a given vector field.

    Signature:

    streamlines_function(field_function, n_seeds=100, seeds=None, seed_center=None,
                         seed_radius=1, method='DOP853', atol=1e-4, rtol=1e-4,
                         metric=None, integration_time=1, integration_steps=10,
                         integration_direction='both', color=(0, 1, 0, 1),
                         color_scalar=None, emission=None, roughness=1,
                         radius=0.1, resolution=8, vmin=None, vmax=None,
                         color_map=None, n_proc=1)

    Parameters
    ----------
    field_function:  Function that is to be integrated. Function has to accept following call signature:
        yy = function(t, xx)
        t:  'time' variable for non-constant functions
        xx:  three-element numpy array of location in cartesian coordinates
        yy:  three-element numpy array representing vector field in cartesian coordinates
    OR:
        yy = function(xx)
        xx:  three-element numpy array of location in cartesian coordinates
        yy:  three-element numpy array representing vector field in cartesian coordinates
    and function will be assumed constant in time

    n_seeds:  Number of randomly distributed seeds within a sphere
        of radius seed_radius centered at seed_center.

    seeds:  Seeds for the streamline tracing of shape (n_seeds, 3).
        Overrides n_seeds.

    seed_center:  Center of the sphere with the seeds.

    seed_radius:  Radius of the sphere with the seeds.

    method:  Integration method for the scipy.integrate.solve_ivp method:
        'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'.

    atol:  Absolute tolerance of the field line tracer.

    rtol:  Relative tolerance of the field line tracer.

    metric:  Metric function that takes a point [x, y, z] and an array
        of shape [3, 3] that has the comkponents g_ij.
        Use 'None' for Cartesian metric.

    integration_time:  Length of the integration time. You need to adapt this according to your
        field strength and box size.

    integration_steps:  Number of integration steps for the field line integration.
        This determines how fine the curve appears.

    integration_direction:  Can be 'forward', 'backward' or 'both' (default).

    color:  rgba values of the form (r, g, b, a) with 0 <= r, g, b, a <= 1, or string,
        e.g. 'red' or character, e.g. 'r', or list of strings/character,
        or [n, 4] array with rgba values or array of the same shape as input array.

    color_scalar:  Scalar function to be used to color the streamlines.
        Set to 'magnitude' to use the vector field's magnitude.

    emission:  Light emission by the streamlines. This overrides 'roughness'.

    roughness:  Texture roughness.

    radius:  Radius of the plotted tube, i.e. line width.

    resolution:  Azimuthal resolution of the tubes in vertices.
        Positive integer > 2.

    vmin, vmax:  Minimum and maximum values for the colormap. If not specify, determine
        from the input arrays.

    color_map:  Color map for the values stored in the array color_scalar.
        These are the same as in matplotlib.

    n_proc:  Number of processors to run the streamline integration on, default 1.

    Returns
    -------
    3d Streamline plot object.

    Examples
    --------
    >>> import numpy as np
    >>> import blendaviz as blt
    >>> def irrational_hopf(t, x):
    >>>     return 1/(1+np.sum(x[0]**2+x[1]**2+x[2]**2))**3 * \
    >>>     np.array([2*(np.sqrt(2)*x[1] - x[0]*x[2]),\
    >>>     -2*(np.sqrt(2)*x[0] + x[1]*x[2]),\
    >>>     (-1 + x[0]**2 +x[1]**2 -x[2]**2)])
    >>> stream = blt.streamlines_function(irrational_hopf, n_seeds=5, integration_time=1000, integration_steps=500)
    """

    # Capture arguments before creating any local variables.
    arguments = dict(locals())

    # Assign parameters to the streamline objects.
    streamlines_return = Streamline3d()
    for key, value in arguments.items():
        setattr(streamlines_return, key, value)
    streamlines_return.plot()
    return streamlines_return



def streamlines_array(x, y, z, u, v, w, n_seeds=100, seeds=None, seed_center=None,
                      seed_radius=1, periodic=None,
                      interpolation='tricubic', method='DOP853', atol=1e-8, rtol=1e-8,
                      metric=None, integration_time=1, integration_steps=10,
                      integration_direction='both',
                      color=(0, 1, 0, 1), color_scalar=None, emission=None, roughness=1,
                      radius=0.1, resolution=8, vmin=None, vmax=None, color_map=None,
                      n_proc=1, time=None):
    """
    Plot streamlines of a given vector field.

    Signature:

    streamlines_array(x, y, z, u, v, w, n_seeds=100, seeds=None, seed_center=None,
                      seed_radius=1, periodic=None,
                      interpolation='tricubic', method='DOP853', atol=1e-8, rtol=1e-8,
                      metric=None, integration_time=1, integration_steps=10,
                      integration_direction='both',
                      color=(0, 1, 0, 1), color_scalar=None, emission=None, roughness=1,
                      radius=0.1, resolution=8, vmin=None, vmax=None, color_map=None,
                      n_proc=1, time=None)

    Parameters
    ----------

    x, y, z:  x, y and z position of the data. These can be 1d arrays of the same length.

    u, v, w:  x, y and z components of the vector field of the shape [nx, ny, nz]

    n_seeds:  Number of randomly distributed seeds within a sphere
        of radius seed_radius centered at seed_center.

    seeds:  Seeds for the streamline tracing of shape [n_seeds, 3].
        Overrides n_seeds.

    seed_center:  Center of the sphere with the seeds.

    seed_radius:  Radius of the sphere with the seeds.

    periodic:  Periodicity array/list for the three directions.
        If true trace streamlines across the boundary and back.

    interpolation:  Interpolation of the vector field.
        'mean': Take the mean of the adjacent grid point.
        'trilinear': Weigh the adjacent grid points according to their distance.
        'tricubic': Use a tricubic spline intnerpolation.

    method:  Integration method for the scipy.integrate.solve_ivp method:
        'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'.

    atol:  Absolute tolerance of the field line tracer.

    rtol:  Relative tolerance of the field line tracer.

    metric:  Metric function that takes a point [x, y, z] and an array
        of shape [3, 3] that has the comkponents g_ij.
        Use 'None' for Cartesian metric.

    integration_time:  Length of the integration time. You need to adapt this according to your
        field strength and box size.

    integration_steps:  Number of integration steps for the field line integration.
        This determines how fine the curve appears.

    integration_direction:  Can be 'forward', 'backward' or 'both' (default).

    color:  rgba values of the form (r, g, b, a) with 0 <= r, g, b, a <= 1, or string,
        e.g. 'red' or character, e.g. 'r', or list of strings/character,
        or [n, 4] array with rgba values or array of the same shape as input array.

    color_scalar:  Scalar array of shape [nx, ny, nz] to be used to color the streamlines.
        Set to 'magnitude' to use the vector field's magnitude.

    emission:  Light emission by the streamlines. This overrides 'roughness'.

    roughness:  Texture roughness.

    radius:  Radius of the plotted tube, i.e. line width.

    resolution:  Azimuthal resolution of the tubes in vertices.
        Positive integer > 2.

    vmin, vmax:  Minimum and maximum values for the colormap. If not specify, determine
        from the input arrays.

    color_map:  Color map for the values stored in the array 'c'.
        These are the same as in matplotlib.

    n_proc:  Number of processors to run the streamline integration on, default 1.

    time:  Float array with the time information of the data.
        Has length nt.

    Returns
    -------
    3d Streamline plot object.

    Examples
    --------
    >>> import numpy as np
    >>> import blendaviz as blt
    >>> x = np.linspace(-4, 4, 100)
    >>> y = np.linspace(-4, 4, 100)
    >>> z = np.linspace(-4, 4, 100)
    >>> xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    >>> u = -yy*np.exp(-np.sqrt(xx**2+yy**2) - zz**2)
    >>> v = xx*np.exp(-np.sqrt(xx**2+yy**2) - zz**2)
    >>> w = np.ones_like(u)*0.1
    >>> stream = blt.streamlines_array(x, y, z, u, v, w, n_seeds=20, integration_time=20, seed_radius=3)
    """

    if not periodic:
        periodic = [False, False, False]

    # Capture arguments before creating any local variables.
    arguments = dict(locals())

    # Assign parameters to the streamline objects.
    streamlines_return = Streamline3dArray()
    for key, value in arguments.items():
        setattr(streamlines_return, key, value)
    streamlines_return.plot()
    return streamlines_return



class Streamline3d(GenericPlot):
    """
    Streamline class containing geometry, parameters and plotting function.
    """

    def __init__(self) -> None:
        """
        Fill members with default values.
        """

        import bpy
        import blendaviz as blt

        super().__init__()

        self.field_function = lambda t, xx: [0., 0., 1.]
        self.n_seeds = 100
        self.seeds = None
        self.seed_center = None
        self.seed_radius = 1
        self.method = 'DOP853'
        self.atol = 1e-4
        self.rtol = 1e-4
        self.metric = None
        self.integration_time = 1
        self.integration_steps = 10
        self.integration_direction = 'both'
        self.color = (0, 1, 0, 1)
        self.color_scalar = None
        self.emission = None
        self.roughness = 1
        self.radius = 0.1
        self.resolution = 8
        self.vmin = None
        self.vmax = None
        self.color_map = None
        self.curve_data = None
        self.curve_object = None
        self.poly_line = None
        self.mesh = None
        self.mesh_material = None
        self.mesh_texture = None
        self.tracers = []
        self.n_proc = 1
        self.deletable_object = None

        # Define the locally used time-independent data and parameters.
        self._field_function = lambda t, xx: [0., 0., 1.]

        # Set the handler function for frame changes (time).
        bpy.app.handlers.frame_change_pre.append(self.time_handler)

        # Add the plot to the stack.
        blt.plot_stack.append(self)


    def plot(self) -> None:
        """
        Plot the streamlines.
        """

        import numpy as np
        import bpy
        from blendaviz import colors

        # Delete existing curves.
        bpy.ops.object.select_all(action='DESELECT')
        if self.mesh is not None:
            bpy.ops.object.select_all(action='DESELECT')
            if self.object_reference_valid(self.mesh):
                self.mesh.select_set(True)
                bpy.context.view_layer.objects.active = self.mesh
                bpy.ops.object.delete()
            del(self.curve_data)
            del(self.curve_object)
            del(self._field_function)
            self.curve_data = None
            self.curve_object = None

        # Delete existing materials.
        if self.mesh_material is not None:
            bpy.ops.object.select_all(action='DESELECT')
            for mesh_material in self.mesh_material:
                bpy.data.materials.remove(mesh_material)
            self.mesh_material = None

        # Prepare the seeds.
        self.__generate_seed_points()

        # Prepare the material colors.
        if isinstance(self.color, list):
            if not any(isinstance(color_index, str) for color_index in self.color):
                self.color = np.array(self.color)
        if isinstance(self.color, np.ndarray):
            if self.color.ndim == 1:
                self.color = self.color[np.newaxis, :]
        if self.color_scalar is None:
            color_rgba = colors.make_rgba_array(self.color, self.n_seeds,
                                                self.color_map, self.vmin, self.vmax)

        self.prepare_field_function()

        # Empty the tracers before calculating new.
        del(self.tracers)
        self.tracers = []

        if self.n_proc == 1:
            # Compute the traces serially
            for tracer_idx in range(self.n_seeds):
                self.tracers.append(self.__tracer(xx=self.seeds[tracer_idx]))
        else:
            # Compute the positions along the streamlines.
            import multiprocessing as mp
            queue = mp.Queue()
            processes = []
            results = []

            for i_proc in range(self.n_proc):
                processes.append(mp.Process(target=self.__tracer_multi,
                                            args=(queue, i_proc, self.n_proc)))
            for i_proc in range(self.n_proc):
                processes[i_proc].start()
            for i_proc in range(self.n_proc):
                results.append(queue.get())
            for i_proc in range(self.n_proc):
                processes[i_proc].join()

            # set the record straight
            result_order = []
            for i_proc in range(self.n_proc):
                result_order.append(results[i_proc][1])
            for i in range(self.n_proc):
                ith_result = result_order.index(i)
                self.tracers.extend(results[ith_result][0]) # tracers


        # Plot the streamlines/tracers.
        self.curve_data = []
        self.curve_object = []
        self.poly_line = []
        self.mesh_material = []
        self.mesh_texture = []
        for tracer_idx in range(self.n_seeds):
            self.curve_data.append(bpy.data.curves.new('DataCurve', type='CURVE'))
            self.curve_data[-1].dimensions = '3D'
            self.curve_object.append(bpy.data.objects.new('ObjCurve', self.curve_data[-1]))

            # Set the origin to the last point.
            self.curve_object[-1].location = tuple((self.tracers[tracer_idx][-1, 0],
                                                    self.tracers[tracer_idx][-1, 1],
                                                    self.tracers[tracer_idx][-1, 2]))

            # Add the rest of the curve.
            self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
            self.poly_line[-1].points.add(self.tracers[tracer_idx].shape[0])
            for param in range(self.tracers[tracer_idx].shape[0]):
                self.poly_line[-1].points[param].co = (self.tracers[tracer_idx][param, 0] - self.tracers[tracer_idx][-1, 0],
                                                       self.tracers[tracer_idx][param, 1] - self.tracers[tracer_idx][-1, 1],
                                                       self.tracers[tracer_idx][param, 2] - self.tracers[tracer_idx][-1, 2],
                                                       0)

            # Add 3d structure.
            self.curve_data[-1].splines.data.bevel_depth = self.radius
            self.curve_data[-1].splines.data.bevel_resolution = self.resolution
            self.curve_data[-1].splines.data.fill_mode = 'FULL'

            # Set the material/color.
            if self.color_scalar is None:
                self.__set_material_color(tracer_idx, color_rgba)
            else:
                self.__set_material_texture(tracer_idx)

            # Link the curve object with the scene.
            bpy.context.scene.collection.objects.link(self.curve_object[-1])

        # Group the curves together.
        bpy.ops.object.select_all(action='DESELECT') # deselect any already selected objects
        for curve_object in self.curve_object[::-1]:
            curve_object.select_set(state=True)
            # curve_object.data.use_uv_as_generated = True
            bpy.context.view_layer.objects.active = curve_object
        # Need to convert to a mesh to retain materials after join.
        bpy.ops.object.convert(target='MESH')
        # Only join if there are multiple objects selected.
        if len(bpy.context.selected_objects) > 1:
            bpy.ops.object.join()
        self.mesh = bpy.context.selected_objects[0]
        self.mesh.select_set(False)

        # Make the grouped meshes the deletable object.
        self.deletable_object = self.mesh

        self.update_globals()

        return 0


    def prepare_field_function(self):
        """
        Prepare the function to be called by the streamline tracing routine.
        """

        import inspect
        import bpy
        import numpy as np

        numargs = len(inspect.signature(self.field_function).parameters)

        # Test if the function takes only one argument.
        if numargs == 1:
            # Replace with function with proper call signature.
            position_function = self.field_function
            self._field_function = lambda t, xx: position_function(xx)
        elif numargs > 3:
            raise TypeError("Function call signature takes too many arguments (max 3: t, x, optional context)")
        else:
            self._field_function = lambda t, xx: self.field_function(bpy.context.scene.frame_float, xx)

        # Evaluate the function.
        testvalue = self._field_function(np.pi, np.random.random(3))
        if (not isinstance(testvalue, np.ndarray)) or (testvalue.size != 3):
            raise TypeError("Function must return a 3-element numpy array")
        return 0


    def __tracer_multi(self, queue, i_proc, n_proc):
        """
        Trace a field starting from xx in any rectilinear coordinate system
        with constant dx, dy and dz and with a given metric.

        Signature:

        tracer(xx=(0, 0, 0))

        Parameters
        ----------
        xx:  Starting point of the field line integration with starting time.
        """

        # Portion up the work given i_proc and n_proc.
        fstep = self.n_seeds/n_proc
        if fstep.is_integer():
            step = int(fstep)
        else:
            step = int(fstep)+1

        start = i_proc*step
        my_chunk = self.seeds[start:start+step] #out-of-range is empty array!
        sub_tracers = []
        for xx in my_chunk:
            my_tracer = self.__tracer(xx)
            sub_tracers.append(my_tracer)

        queue.put((sub_tracers, i_proc, n_proc))
        return 0


    def __tracer(self, xx=(0, 0, 0)):
        """
        Trace a field starting from xx in any rectilinear coordinate system
        with constant dx, dy and dz and with a given metric.

        Signature:

        tracer(xx=(0, 0, 0))

        Parameters
        ----------
        xx:  Starting point of the field line integration with starting time.
        """

        import numpy as np
        from scipy.integrate import solve_ivp

        time = np.linspace(0, self.integration_time, self.integration_steps)
        if self.integration_direction == 'backward':
            time = -time

        if not self.metric:
            self.metric = lambda xx: np.eye(3)

        # Set up the ode solver.
        tracers = solve_ivp(self._field_function, (time[0], time[-1]), xx,
                            t_eval=time, rtol=self.rtol, atol=self.atol,
                            method=self.method).y.T

        # In case of forward and backward field integration trace backward.
        if self.integration_direction == 'both':
            time = -time
            backtracers = solve_ivp(self._field_function, (time[0], time[-1]), xx,
                                    t_eval=time, rtol=self.rtol, atol=self.atol,
                                    method=self.method).y.T
            # Glue the forward and backward field tracers together.
            tracers = np.vstack([backtracers[::-1, :], tracers[1:, :]])

        # Delete points outside the domain.
        tracers = self.delete_outside_points(tracers)

        return tracers


    def delete_outside_points(self, tracers):
        """
        Dummy function for deleting any points of the tracer that lie outside the domain.

        Signature:

        delete_outside_points(tracers)

        Parameters
        ----------
        tracers:  Field line tracer array.
        """

        return tracers


    def __set_material_color(self, idx, color_rgba):
        """
        Set the mesh material color.

        Signature:

        __set_material(idx, color_rgba)

        Parameters
        ----------
        idx:  Index of the material.

        color_rgba:  The rgba values of the colors to be used.
        """

        import bpy
        import numpy as np

        # Deterimne if we need a list of materials, i.e. for every streamline one.
        if any([isinstance(self.emission, np.ndarray),
                isinstance(self.roughness, np.ndarray)]):
            list_material = True
            # Make sure the color is also an array.
            if not isinstance(color_rgba, np.ndarray):
                color_rgba = np.repeat([color_rgba, ], self.n_seeds, axis=0)
        else:
            list_material = False

        # Transform single values to arrays.
        if list_material:
            if not isinstance(self.roughness, np.ndarray):
                self.roughness = np.ones(self.n_seeds)*self.roughness
            if self.emission is not None:
                if not isinstance(self.emission, np.ndarray):
                    self.emission = np.ones(self.n_seeds)*self.emission

        # Set the material.
        if list_material:
            self.mesh_material.append(bpy.data.materials.new('material'))
            self.curve_object[idx].active_material = self.mesh_material[idx]
        else:
            if idx == 0:
                self.mesh_material.append(bpy.data.materials.new('material'))
                self.mesh_material[0].diffuse_color = color_rgba
            self.curve_object[idx].active_material = self.mesh_material[0]

        # Set the diffusive color.
        if list_material:
            self.mesh_material[idx].diffuse_color = color_rgba[idx]
        else:
            self.mesh_material[0].diffuse_color = color_rgba

        # Set the material roughness.
        if list_material:
            if isinstance(self.roughness, np.ndarray):
                self.mesh_material[idx].roughness = self.roughness[idx]
            else:
                self.mesh_material[idx].roughness = self.roughness
        elif idx == 0:
            self.mesh_material[0].roughness = self.roughness

        # Set the material emission.
        if self.emission is not None:
            if list_material:
                self.mesh_material[idx].use_nodes = True
                node_tree = self.mesh_material[idx].node_tree
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

                # Change the input of the ouput node to emission.
                node_tree.links.new(node_emission.outputs['Emission'],
                                    output_node.inputs['Surface'])

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

                # Change the input of the ouput node to emission.
                node_tree.links.new(node_emission.outputs['Emission'],
                                    output_node.inputs['Surface'])

                # Adapt emission and color.
                node_emission.inputs['Color'].default_value = color_rgba
                if isinstance(self.emission, np.ndarray):
                    node_emission.inputs['Strength'].default_value = self.emission
                else:
                    node_emission.inputs['Strength'].default_value = self.emission


    def __set_material_texture(self, tracer_idx):
        """
        Set the mesh material texture.

        Signature:

        __set_material_texture(tracer_idx)

        Parameters
        ----------
        tracer_idx:  Index of the tracer.
        """

        import bpy
        import numpy as np
        from matplotlib import cm

        # Compute the scalar values along the streamline.
        scalar_values = self.set_texture_scalar_values(tracer_idx)

        # Prepare the texture.
        mesh_image = bpy.data.images.new('ImageMesh', self.tracers[tracer_idx].shape[0], 1)
        pixels = np.array(mesh_image.pixels)

        # Assign the RGBa values to the pixels.
        if self.color_map is None:
            self.color_map = cm.viridis
        pixels[0::4] = self.color_map((scalar_values - self.vmin)/(self.vmax - self.vmin))[:, 0]
        pixels[1::4] = self.color_map((scalar_values - self.vmin)/(self.vmax - self.vmin))[:, 1]
        pixels[2::4] = self.color_map((scalar_values - self.vmin)/(self.vmax - self.vmin))[:, 2]
        pixels[3::4] = 1
        mesh_image.pixels[:] = np.swapaxes(pixels.reshape([scalar_values.shape[0],
                                                           1, 4]), 0, 1).flatten()[:]

        # Create the material.
        self.mesh_material.append(bpy.data.materials.new('material'))
        self.curve_object[tracer_idx].active_material = self.mesh_material[tracer_idx]

        # Assign the texture to the material.
        self.mesh_material[tracer_idx].use_nodes = True
        self.mesh_texture.append(self.mesh_material[tracer_idx].node_tree.nodes.new('ShaderNodeTexImage'))
        self.mesh_texture[tracer_idx].extension = 'EXTEND'
        self.mesh_texture[tracer_idx].image = mesh_image
        links = self.mesh_material[tracer_idx].node_tree.links
        links.new(self.mesh_texture[tracer_idx].outputs[0],
                  self.mesh_material[tracer_idx].node_tree.nodes.get("Principled BSDF").inputs[0])


    def set_texture_scalar_values(self, tracer_idx):
        """
        Find the scalar values for generating the texture along the streamlines.

        Signature:

        set_texture_scalar_values(tracer_idx)

        Parameters
        ----------
        tracer_idx:  Index of the tracer.
        """

        import numpy as np

        scalar_values = np.zeros(self.tracers[tracer_idx].shape[0])
        if self.color_scalar == 'magnitude':
            for idx in range(self.tracers[tracer_idx].shape[0]):
                scalar_values[idx] = np.sqrt(np.sum(self._field_function(0, self.tracers[tracer_idx][idx, :])**2))
        else:
            for idx in range(self.tracers[tracer_idx].shape[0]):
                scalar_values[idx] = np.sqrt(np.sum(self.color_scalar(self.tracers[tracer_idx][idx, :])**2))

        # Set vmin and vmax from scalar values if not already set.
        if not isinstance(self.vmin, (int, float)):
            self.vmin = scalar_values.min()
        if not isinstance(self.vmax, (int, float)):
            self.vmax = scalar_values.max()

        return scalar_values


    def __generate_seed_points(self):
        """
        Generate the seed points for the streamline tracer.
        Generates a random 3D unit vector (direction) with a uniform spherical distribution,
        and a uniformly distributed radius.
        This means points are weighted towards the center!
        Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
        """

        import numpy as np

        if isinstance(self.seeds, np.ndarray):
            if self.seeds.ndim == 1:
                self.n_seeds = 1
                self.seeds = np.expand_dims(self.seeds, axis=0)
            self.n_seeds = self.seeds.shape[0]
        else:
            if self.seed_center is None:  # make a center if not exists
                if hasattr(self, 'x'):
                    self.seed_center = np.array([self.x.max() + self.x.min(),
                                                 self.y.max() + self.y.min(),
                                                 self.z.max() + self.z.min()])/2
                else:
                    self.seed_center = np.zeros(3)
            phi = np.random.uniform(0, 2*np.pi, self.n_seeds)
            costheta = np.random.uniform(-1, 1, self.n_seeds)
            theta = np.arccos(costheta)
            radius = self.seed_radius*np.cbrt(np.random.uniform(0, 1, self.n_seeds))
            x = radius*np.sin(theta)*np.cos(phi) + self.seed_center[0]
            y = radius*np.sin(theta)*np.sin(phi) + self.seed_center[1]
            z = radius*np.cos(theta) + self.seed_center[2]
            self.seeds = np.array([x, y, z]).T

        return 0


    def time_handler(self, scene, depsgraph):
        """
        Function to be called whenever any Blender animation functions are used.
        Updates the plot according to the function specified.
        """

        import inspect

        numargs = len(inspect.signature(self.field_function).parameters)
        if numargs == 2:
            self.plot()



class Streamline3dArray(Streamline3d):
    """
    Derived streamline class for field function given as data array.
    """

    def __init__(self) -> None:
        """
        Fill members with default values.
        """

        import bpy

        super().__init__()

        self.x = None
        self.y = None
        self.z = None
        self.u = None
        self.v = None
        self.w = None
        self.time = None
        self.time_index = 0
        self.periodic = [False, False, False]
        self.interpolation = 'tricubic'

        # Define the locally used time-independent data and parameters.
        self._x = None
        self._y = None
        self._z = None
        self._u = None
        self._v = None
        self._w = None

        # Set the handler function for frame changes (time).
        bpy.app.handlers.frame_change_pre.append(self.time_handler)


    def prepare_field_function(self):
        """
        Prepare the function to be called by the streamline tracing routine.
        Sets: self.field_function
        """

        import numpy as np
        import bpy

        # Check if there is any time array.
        if self.time is not None:
            if not isinstance(self.time, np.ndarray):
                print("Error: time is not a valid array.")
                return -1
            if self.time.ndim != 1:
                print("Error: time array must be 1d.")
                return -1
            # Determine the time index.
            self.time_index = np.argmin(abs(bpy.context.scene.frame_float - self.time))
        else:
            self.time_index = 0

        # Check the validity of the input arrays.
        if not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray) \
           or not isinstance(self.z, np.ndarray):
            print("Error: x OR y OR z array invalid.")
            return -1
        if not (self.x.shape == self.y.shape == self.z.shape) and \
               (self.u.shape == self.v.shape == self.w.shape):
            print("Error: input array shapes invalid.")
            return -1

        # Point the local variables to the correct arrays.
        arrays_with_time_list = ['x', 'y', 'z', 'u', 'v', 'w']
        for array_with_time in arrays_with_time_list:
            array_value = getattr(self, array_with_time)
            if array_value.ndim in (1, 3):
                setattr(self, '_' + array_with_time, array_value)
            else:
                setattr(self, '_' + array_with_time, array_value[..., self.time_index])

        if self.interpolation == 'tricubic':
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    from eqtools.trispline import Spline
            except (ImportError, ModuleNotFoundError) as e:
                print(f'Warning: Could not import eqtools.trispline.Spline for '
                      f'tricubic interpolation: {e}')
                print('Warning: Falling back to trilinear interpolation.')
                self.interpolation = 'trilinear'

        # Set up the splines for the tricubic interpolation.
        if self.interpolation == 'tricubic':
            splines = []
            splines.append(Spline(self._z, self._y, self._x, np.swapaxes(self._u, 0, 2)))
            splines.append(Spline(self._z, self._y, self._x, np.swapaxes(self._v, 0, 2)))
            splines.append(Spline(self._z, self._y, self._x, np.swapaxes(self._w, 0, 2)))
        else:
            splines = None

        # Redefine the derivative y for the scipy ode integrator using the given parameters.
        if self.interpolation in ('mean', 'trilinear'):
            self._field_function = lambda t, xx: self.__vec_int(xx)
        if self.interpolation == 'tricubic':
            field_x = splines[0]
            field_y = splines[1]
            field_z = splines[2]
            self._field_function = lambda t, xx: self.__trilinear_func(xx, field_x, field_y, field_z)

        return 0


    def set_texture_scalar_values(self, tracer_idx):
        """
        Find the scalar values for generating the texture along the streamlines.

        Signature:

        set_texture_scalar_values(tracer_idx)

        tracer_idx:  Index of the tracer.
        """

        import numpy as np

        scalar_values = np.zeros(self.tracers[tracer_idx].shape[0])
        if isinstance(self.color_scalar, str):
            if self.color_scalar == 'magnitude':
                for idx in range(self.tracers[tracer_idx].shape[0]):
                    scalar_values[idx] = np.sqrt(np.sum(self._field_function(0, self.tracers[tracer_idx][idx, :])**2))
        else:
            from scipy.interpolate import RegularGridInterpolator
            # Prepare the interpolation function.
            scalar_interpolation = RegularGridInterpolator((self._x, self._y, self._z), self.color_scalar)
            for idx in range(self.tracers[tracer_idx].shape[0]):
                scalar_values[idx] = scalar_interpolation(self.tracers[tracer_idx][idx, :])

        # Find the min and max values of the sclara field, if there is any.
        if self.color_scalar is None:
            self.vmin = 0
            self.vmax = 1
        else:
            if isinstance(self.color_scalar, str):
                if not isinstance(self.vmin, (int, float)):
                    self.vmin = np.sqrt(np.min(self.u**2 + self.v**2 + self.w**2))
                if not isinstance(self.vmax, (int, float)):
                    self.vmax = np.sqrt(np.max(self.u**2 + self.v**2 + self.w**2))
            else:
                if not isinstance(self.vmin, (int, float)):
                    self.vmin = self.color_scalar.min()
                if not isinstance(self.vmax, (int, float)):
                    self.vmax = self.color_scalar.max()

        return scalar_values


    def __trilinear_func(self, xx, field_x, field_y, field_z,):
        """
        Trilinear spline interpolation like eqtools.trispline.Spline
        but return 0 if the point lies outside the box.

        Signature:

        trilinear_func(xx, field_x, field_y, field_z,)

        Parameters
        ----------
        xx:  The xyz coordinates of the point to interpolate the data.

        field_xyz:  The Spline objects for the velocity fields.
        """

        import numpy as np

        # Determine some parameters.
        Ox = self._x.min()
        Oy = self._y.min()
        Oz = self._z.min()
        Lx = self._x.max()
        Ly = self._y.max()
        Lz = self._z.max()

        out_of_bounds = np.any([xx[0] < Ox, xx[0] > Lx,
                                 xx[1] < Oy, xx[1] > Ly,
                                 xx[2] < Oz, xx[2] > Lz])
        if out_of_bounds:
            field = np.zeros(3)
            if self.periodic[0]:
                field[0] = (xx[0] - Ox) % (Lx - Ox) + Ox
            if self.periodic[1]:
                field[1] = (xx[1] - Oy) % (Ly - Oy) + Oy
            if self.periodic[2]:
                field[2] = (xx[2] - Oz) % (Lz - Oz) + Oz
            return field

        return np.array([field_x.ev(xx[2], xx[1], xx[0]),
                         field_y.ev(xx[2], xx[1], xx[0]),
                         field_z.ev(xx[2], xx[1], xx[0])])[:, 0]


    def __vec_int(self, xx):
        """
        Interpolates the vector field around position xx.

        Signature:

        vec_int(xx)

        Parameters
        ----------
        xx:  Position vector around which field will be interpolated.

        interpolation: The type of interpolation that is to be performed,
            'mean', 'trilinear' or 'tricubic'
        """

        import numpy as np

        def surrounding_indices(x, origin, d, n, periodic):
            '''
            Find the surrounding indices for a position vector.
            '''

            i = (x - origin) / d

            i0 = int(np.floor(i))
            i1 = i0 + 1

            if periodic:
                i0 %= n
                i1 %= n
            else:
                i0 = np.clip(i0, 0, n-1)
                i1 = np.clip(i1, 0, n-1)

            return i, np.array([i0, i1])

        # Determine some parameters.
        Ox = self._x.min()
        Oy = self._y.min()
        Oz = self._z.min()
        dx = self._x[1] - self._x[0]
        dy = self._y[1] - self._y[0]
        dz = self._z[1] - self._z[0]
        nx = np.size(self._x)
        ny = np.size(self._y)
        nz = np.size(self._z)

        if self.interpolation in ('mean', 'trilinear'):
            # Use temporary pointers to the data. This is convenient for the periodic case.
            u = self._u
            v = self._v
            w = self._w

            i, ii = surrounding_indices(xx[0], Ox, dx, nx, self.periodic[0])
            j, jj = surrounding_indices(xx[1], Oy, dy, ny, self.periodic[1])
            k, kk = surrounding_indices(xx[2], Oz, dz, nz, self.periodic[2])

        # Interpolate the field.
        if self.interpolation == 'mean':
            sub_field = np.array([
                [
                    [
                        [u[ii[0], jj[0], kk[0]], u[ii[0], jj[0], kk[1]]],
                        [u[ii[0], jj[1], kk[0]], u[ii[0], jj[1], kk[1]]],
                    ],
                    [
                        [u[ii[1], jj[0], kk[0]], u[ii[1], jj[0], kk[1]]],
                        [u[ii[1], jj[1], kk[0]], u[ii[1], jj[1], kk[1]]],
                    ],
                ],
                [
                    [
                        [v[ii[0], jj[0], kk[0]], v[ii[0], jj[0], kk[1]]],
                        [v[ii[0], jj[1], kk[0]], v[ii[0], jj[1], kk[1]]],
                    ],
                    [
                        [v[ii[1], jj[0], kk[0]], v[ii[1], jj[0], kk[1]]],
                        [v[ii[1], jj[1], kk[0]], v[ii[1], jj[1], kk[1]]],
                    ],
                ],
                [
                    [
                        [w[ii[0], jj[0], kk[0]], w[ii[0], jj[0], kk[1]]],
                        [w[ii[0], jj[1], kk[0]], w[ii[0], jj[1], kk[1]]],
                    ],
                    [
                        [w[ii[1], jj[0], kk[0]], w[ii[1], jj[0], kk[1]]],
                        [w[ii[1], jj[1], kk[0]], w[ii[1], jj[1], kk[1]]],
                    ],
                ],
            ])
            return np.mean(np.array(sub_field), axis=(1, 2, 3))

        if self.interpolation == 'trilinear':
            fx = i - np.floor(i)
            fy = j - np.floor(j)
            fz = k - np.floor(k)

            wx = np.array([1 - fx, fx])
            wy = np.array([1 - fy, fy])
            wz = np.array([1 - fz, fz])

            weight = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
            sub_field = np.array([
                [
                    [
                        [u[ii[0], jj[0], kk[0]], u[ii[0], jj[0], kk[1]]],
                        [u[ii[0], jj[1], kk[0]], u[ii[0], jj[1], kk[1]]],
                    ],
                    [
                        [u[ii[1], jj[0], kk[0]], u[ii[1], jj[0], kk[1]]],
                        [u[ii[1], jj[1], kk[0]], u[ii[1], jj[1], kk[1]]],
                    ],
                ],
                [
                    [
                        [v[ii[0], jj[0], kk[0]], v[ii[0], jj[0], kk[1]]],
                        [v[ii[0], jj[1], kk[0]], v[ii[0], jj[1], kk[1]]],
                    ],
                    [
                        [v[ii[1], jj[0], kk[0]], v[ii[1], jj[0], kk[1]]],
                        [v[ii[1], jj[1], kk[0]], v[ii[1], jj[1], kk[1]]],
                    ],
                ],
                [
                    [
                        [w[ii[0], jj[0], kk[0]], w[ii[0], jj[0], kk[1]]],
                        [w[ii[0], jj[1], kk[0]], w[ii[0], jj[1], kk[1]]],
                    ],
                    [
                        [w[ii[1], jj[0], kk[0]], w[ii[1], jj[0], kk[1]]],
                        [w[ii[1], jj[1], kk[0]], w[ii[1], jj[1], kk[1]]],
                    ],
                ],
            ])
            return np.sum(np.array(sub_field)*weight, axis=(1, 2, 3))/np.sum(weight)

        # If the point lies outside the domain, return 0.
        if (ii[0] < -1) or (ii[1] > nx) or (jj[0] < -1) or (jj[1] > ny) \
            or (kk[0] < -1) or (kk[1] > nz):
            return np.zeros([0, 0, 0])

        return -1

    def delete_outside_points(self, tracers):
        """
        Delete any points of the tracer that lie outside the domain.

        Signature:

        delete_outside_points(tracers)

        Parameters
        ----------
        tracers:  Field line tracer array.
        """

        import numpy as np

        # Determine some parameters.
        Ox = self._x.min()
        Oy = self._y.min()
        Oz = self._z.min()
        Lx = self._x.max() - self._x.min()
        Ly = self._y.max() - self._y.min()
        Lz = self._z.max() - self._z.min()

        # Remove points that lie outside the domain and interpolation on the boundary.
        cut_mask = ((tracers[:, 0] > Ox+Lx) + \
                    (tracers[:, 0] < Ox))*(not self.periodic[0]) + \
                   ((tracers[:, 1] > Oy+Ly) + \
                    (tracers[:, 1] < Oy))*(not self.periodic[1]) + \
                   ((tracers[:, 2] > Oz+Lz) + \
                    (tracers[:, 2] < Oz))*(not self.periodic[2])
        if np.sum(cut_mask) > 0:
            # Find the first point that lies outside.
            idx_outside = np.min(np.where(cut_mask))
            # Interpolate.
            p0 = tracers[idx_outside-1, :]
            p1 = tracers[idx_outside, :]
            lam = np.zeros([6])
            if p0[0] == p1[0]:
                lam[0] = np.inf
                lam[1] = np.inf
            else:
                lam[0] = (Ox + Lx - p0[0])/(p1[0] - p0[0])
                lam[1] = (Ox - p0[0])/(p1[0] - p0[0])
            if p0[1] == p1[1]:
                lam[2] = np.inf
                lam[3] = np.inf
            else:
                lam[2] = (Oy + Ly - p0[1])/(p1[1] - p0[1])
                lam[3] = (Oy - p0[1])/(p1[1] - p0[1])
            if p0[2] == p1[2]:
                lam[4] = np.inf
                lam[5] = np.inf
            else:
                lam[4] = (Oz + Lz - p0[2])/(p1[2] - p0[2])
                lam[5] = (Oz - p0[2])/(p1[2] - p0[2])
            lam_min = np.min(lam[lam >= 0])
            if abs(lam_min) == np.inf:
                lam_min = 0
            tracers[idx_outside, :] = p0 + lam_min*(p1-p0)
            # We don't want to cut the interpolated point (was first point outside).
            cut_mask[idx_outside] = False
            cut_mask[idx_outside+1:] = True
            # Remove outside points.
            tracers = tracers[~cut_mask, :].copy()

        return tracers


    def time_handler(self, scene, depsgraph):
        """
        Function to be called whenever any Blender animation functions are used.
        Updates the plot according to the function specified.
        """

        if self.time is not None:
            self.plot()


    def update_globals(self) -> None:
        """
        Update the extrema and lights.
        """

        import blendaviz as blt

        self.update_extrema_x()
        self.update_extrema_y()
        self.update_extrema_z()

        if blt.house_keeping.box is None:
            blt.house_keeping.box = blt.bounding_box()
        else:
            blt.house_keeping.box.get_extrema()
            blt.house_keeping.box.plot()

        # Add some light.
        blt.adjust_lights()
