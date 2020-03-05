# streamlines3d.py
"""
Contains routines to generate and plot streamlines.

Created on Thu Jan 09 16:33:00 2020

@authors: Simon Candelaresi
"""


'''
Test:
import numpy as np
import importlib
import sys
sys.path.append('~/codes/blendaviz')
import blendaviz as blt
importlib.reload(blt)
importlib.reload(blt.streamlines3d)
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
z = np.linspace(-4, 4, 100)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
u = -yy*np.exp(-np.sqrt(xx**2+yy**2) - zz**2)
v = xx*np.exp(-np.sqrt(xx**2+yy**2) - zz**2)
w = np.ones_like(u)*0.1
stream = blt.streamlines(x, y, z, u, v, w, seeds=8, integration_time=100, integration_steps=10)
'''

# TODO:
# - 1) Color and material options.
# - 2) Interpolation on non-equidistant grids.
# - 4) Implement periodic domains.
# - 5) Different kinds of seeds, like spherical with radius and origin.

def streamlines(x, y, z, u, v, w, seeds=100, periodic=None,
                interpolation='tricubic', method='dop853', atol=1e-8, rtol=1e-8,
                metric=None, integration_time=1, integration_steps=10,
                integration_direction='both',
                color=(0, 1, 0, 1), emission=None, roughness=1,
                radius=0.1, resolution=8, vmin=None, vmax=None, color_map=None):
    """
    Plot streamlines of a given vector field.

    call signature:

    streamlines(x, y, z, u, v, w, seeds=100, periodic=None,
                interpolation='tricubic', method='dop853', atol=1e-8, rtol=1e-8,
                metric=None, integration_time=1, integration_steps=10,
                integration_direction='both',
                color=(0, 1, 0, 1), emission=None, roughness=1,
                radius=0.1, resolution=8, vmin=None, vmax=None, color_map=None)

    Keyword arguments:
    *x, y, z*:
      x, y and z position of the data. These can be 1d arrays of the same length.

    *u, v, w*
      x, y and z components of the vector field of the shape [nx, ny, nz]

    *seeds*
      Seeds for the streamline tracing.
      If single number, generate randomly distributed seeds withing x, y, z.
      If array of size [n_seeds, 3] use for the seeds positions.

    *periodic*:
      Periodicity array/list for the three directions.
      If true trace streamlines across the boundary and back.

    *interpolation*:
       Interpolation of the vector field.
       'mean': Take the mean of the adjacent grid point.
       'trilinear': Weigh the adjacent grid points according to their distance.
       'tricubic': Use a tricubic spline intnerpolation.

    *method*:
        Integration method for the scipy.integrate.ode method.

    *atol*:
      Absolute tolerance of the field line tracer.

    *rtol*:
      Relative tolerance of the field line tracer.

    *metric*:
        Metric function that takes a point [x, y, z] and an array
        of shape [3, 3] that has the comkponents g_ij.
        Use 'None' for Cartesian metric.

    *integration_time*:
      Length of the integration time. You need to adapt this according to your
      field strength and box size.

    *integration_steps*:
      Number of integration steps for the field line integration.
      This determines how fine the curve appears.

    *integration_direction:
      Can be 'forward', 'backward' or 'both' (default).

    *color*:
      rgba values of the form (r, g, b, a) with 0 <= r, g, b, a <= 1, or string,
      e.g. 'red' or character, e.g. 'r', or list of strings/character,
      or [n, 4] array with rgba values or array of the same shape as input array.

    *emission*
      Light emission by the streamlines. This overrides 'roughness'.

    *roughness*:
      Texture roughness.

    *radius*:
      Radius of the plotted tube, i.e. line width.

    *resolution*:
      Azimuthal resolution of the tubes in vertices.
      Positive integer > 2.

    *vmin, vmax*:
      Minimum and maximum values for the colormap. If not specify, determine
      from the input arrays.

    *color_map*:
      Color map for the values stored in the array 'c'.
      These are the same as in matplotlib.
    """

    import inspect

    if not periodic:
        periodic = [False, False, False]

    # Assign parameters to the streamline objects.
    streamlines_return = Streamline3d()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(streamlines_return, argument, argument_dict[argument])
    streamlines_return.plot()
    return streamlines_return


class Streamline3d(object):
    """
    Streamline class containing geometry, parameters and plotting function.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        import numpy as np

        self.x = np.array([0, 1])
        self.y = np.array([0, 1])
        self.z = np.array([0, 1])
        self.u = np.array([0, 1])
        self.v = np.array([0, 1])
        self.w = np.array([0, 1])
        self.seeds = 100
        self.periodic = [False, False, False]
        self.interpolation = 'tricubic'
        self.method = 'dop853'
        self.atol = 1e-8
        self.rtol = 1e-8
        self.metric = None
        self.integration_time = 1
        self.integration_steps = 10
        self.integration_direction = 'both'
        self.color = (0, 1, 0, 1)
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


    def plot(self):
        """
        Plot the streamlines.
        """

        import bpy
        import numpy as np
        from . import colors

        # Check the validity of the input arrays.
        if not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray) \
           or not isinstance(self.z, np.ndarray):
            print("Error: x OR y OR z array invalid.")
            return -1
        if not (self.x.shape == self.y.shape == self.z.shape) and \
               (self.u.shape == self.v.shape == self.w.shape):
            print("Error: input array shapes invalid.")
            return -1

        # Delete existing curve.
        if not self.mesh is None:
            bpy.ops.object.select_all(action='DESELECT')
            self.mesh.select_set(True)
            bpy.ops.object.delete()
            self.curve_data = None
            self.curve_object = None

        # Delete existing materials.
        if not self.mesh_material is None:
            bpy.ops.object.select_all(action='DESELECT')
            for mesh_material in self.mesh_material:
                bpy.data.materials.remove(mesh_material)
            self.mesh_material = None

        # Prepare the seeds.
        if isinstance(self.seeds, int):
            self.seeds = np.random.random([self.seeds, 3])
            self.seeds[:, 0] = self.seeds[:, 0]*(self.x.max()-self.x.min()) + self.x.min()
            self.seeds[:, 1] = self.seeds[:, 1]*(self.y.max()-self.y.min()) + self.y.min()
            self.seeds[:, 2] = self.seeds[:, 2]*(self.z.max()-self.z.min()) + self.z.min()
        if not isinstance(self.seeds, np.ndarray):
            print("Error: seeds are not valid.")
            return -1

        # Prepare the material colors.
        color_rgba = colors.make_rgba_array(self.color, self.seeds.shape[0],
                                            self.color_map, self.vmin, self.vmax)

        # Set up the field line integration.
        if self.interpolation == 'tricubic':
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    from eqtools.trispline import Spline
            except:
                print('Warning: Could not import eqtools.trispline.Spline for \
                       tricubic interpolation.\n')
                print('Warning: Fall back to trilinear.')
                self.interpolation = 'trilinear'
        # Set up the splines for the tricubic interpolation.
        if self.interpolation == 'tricubic':
            splines = []
            splines.append(Spline(self.z, self.y, self.x, np.swapaxes(self.u, 0, 2)))
            splines.append(Spline(self.z, self.y, self.x, np.swapaxes(self.v, 0, 2)))
            splines.append(Spline(self.z, self.y, self.x, np.swapaxes(self.w, 0, 2)))
        else:
            splines = None

        # Compute the positions along the streamlines.
        tracers = []
        for tracer_idx in range(self.seeds.shape[0]):
            tracers.append(self.__tracer(xx=self.seeds[tracer_idx], splines=splines))

        # Plot the streamlines/tracers.
        self.curve_data = []
        self.curve_object = []
        self.poly_line = []
        self.mesh_material = []
        for tracer_idx in range(self.seeds.shape[0]):
            self.curve_data.append(bpy.data.curves.new('DataCurve', type='CURVE'))
            self.curve_data[-1].dimensions = '3D'
            self.curve_object.append(bpy.data.objects.new('ObjCurve', self.curve_data[-1]))

            # Set the origin to the last point.
            self.curve_object[-1].location = tuple((tracers[tracer_idx][-1, 0],
                                                    tracers[tracer_idx][-1, 1],
                                                    tracers[tracer_idx][-1, 2]))

            # Add the rest of the curve.
            self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
            self.poly_line[-1].points.add(tracers[tracer_idx].shape[0])
            for param in range(tracers[tracer_idx].shape[0]):
                self.poly_line[-1].points[param].co = (tracers[tracer_idx][param, 0] - tracers[tracer_idx][-1, 0],
                                                       tracers[tracer_idx][param, 1] - tracers[tracer_idx][-1, 1],
                                                       tracers[tracer_idx][param, 2] - tracers[tracer_idx][-1, 2],
                                                       0)

            # Add 3d structure.
            self.curve_data[-1].splines.data.bevel_depth = self.radius
            self.curve_data[-1].splines.data.bevel_resolution = self.resolution
            self.curve_data[-1].splines.data.fill_mode = 'FULL'

            # Set the material/color.
            self.__set_material(tracer_idx, color_rgba)

            # Link the curve object with the scene.
            bpy.context.scene.collection.objects.link(self.curve_object[-1])

        # Group the curves together.
        for curve_object in self.curve_object[::-1]:
            curve_object.select_set(state=True)
            bpy.context.view_layer.objects.active = curve_object
        bpy.ops.object.join()
        self.mesh = bpy.context.selected_objects[0]
        self.mesh.select_set(False)

        return 0


    def __tracer(self, xx=(0, 0, 0), splines=None):
        """
        Trace a field starting from xx in any rectilinear coordinate system
        with constant dx, dy and dz and with a given metric.

        call signature:

          tracer(xx=(0, 0, 0), metric=None, splines=None):

        Keyword arguments:

        *xx*:
          Starting point of the field line integration with starting time.

        *splines*:
            Spline interpolation functions for the tricubic interpolation.
            This can speed up the calculations greatly for repeated streamline tracing
            on the same data.
            Accepts a list of the spline functions for the three vector components.
        """

        import numpy as np
        from scipy.integrate import ode
        from scipy.integrate import solve_ivp

        time = np.linspace(0, self.integration_time, self.integration_steps)
        field_sign = +1
        if self.integration_direction == 'backward':
            field_sign = -1

        # Determine some parameters.
        Ox = self.x.min()
        Oy = self.y.min()
        Oz = self.z.min()
        Lx = self.x.max()-self.x.min()
        Ly = self.y.max()-self.y.min()
        Lz = self.z.max()-self.z.min()

        if not self.metric:
            self.metric = lambda xx: np.eye(3)

        # Redefine the derivative y for the scipy ode integrator using the given parameters.
        if (self.interpolation == 'mean') or (self.interpolation == 'trilinear'):
            odeint_func = lambda t, xx: self.__vec_int(xx)*field_sign
        if self.interpolation == 'tricubic':
            field_x = splines[0]
            field_y = splines[1]
            field_z = splines[2]
            odeint_func = lambda t, xx: self.__trilinear_func(xx, field_x, field_y, field_z)*field_sign

        # Set up the ode solver.
        methods_ode = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
        methods_ivp = ['RK45', 'RK23', 'Radau', 'BDF', 'LSODA']
        if self.method in methods_ode:
            solver = ode(odeint_func, jac=self.metric)
            solver.set_initial_value(xx, time[0])
            solver.set_integrator(self.method, rtol=self.rtol, atol=self.atol)
            tracers = np.zeros([len(time), 3])
            tracers[0, :] = xx
            for i, t in enumerate(time[1:]):
                tracers[i+1, :] = solver.integrate(t)
        if self.method in methods_ivp:
            tracers = solve_ivp(odeint_func, (time[0], time[-1]), xx,
                                t_eval=time, rtol=self.rtol, atol=self.atol,
                                jac=self.metric, method=self.method).y.T

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

        # In case of forward and backward field integration trace backward.
        if self.integration_direction == 'both':
            self.integration_direction = 'backward'
            tracers_backward = self.__tracer(xx=xx, splines=splines)
            self.integration_direction = 'both'
            # Glue the forward and backward field tracers together.
            tracers = np.vstack([tracers_backward[::-1, :], tracers[1:, :]])

        return tracers


    def __trilinear_func(self, xx, field_x, field_y, field_z,):
        """
        Trilinear spline interpolation like eqtools.trispline.Spline
        but return 0 if the point lies outside the box.

        call signature:

        trilinear_func(xx, field_x, field_y, field_z,)

        Keyword arguments:

        *xx*:
          The zyx coordinates of the point to interpolate the data.

        *field_xyz*:
          The Spline objects for the velocity fields.
        """

        import numpy as np

        # Determine some parameters.
        Ox = self.x.min()
        Oy = self.y.min()
        Oz = self.z.min()
        Lx = self.x.max()
        Ly = self.y.max()
        Lz = self.z.max()

        if (xx[0] < Ox) + (xx[0] > Ox + Lx) + \
           (xx[1] < Oy) + (xx[1] > Oy + Ly) + \
           (xx[2] < Oz) + (xx[2] > Oz + Lz):
            return np.zeros(3)
        return np.array([field_x.ev(xx[2], xx[1], xx[0]),
                         field_y.ev(xx[2], xx[1], xx[0]),
                         field_z.ev(xx[2], xx[1], xx[0])])[:, 0]


    def __vec_int(self, xx):
        """
        Interpolates the vector field around position xx.

        call signature:

            vec_int(xx)

        Keyword arguments:

        *xx*:
          Position vector around which field will be interpolated.
        """

        import numpy as np

        # Determine some parameters.
        Ox = self.x.min()
        Oy = self.y.min()
        Oz = self.z.min()
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        nx = np.size(self.x)
        ny = np.size(self.y)
        nz = np.size(self.z)

        if (self.interpolation == 'mean') or (self.interpolation == 'trilinear'):
            # Find the adjacent indices.
            i = (xx[0]-Ox)/dx
            ii = np.array([int(np.floor(i)), int(np.ceil(i))])
            if not self.periodic[0]:
                if i < 0:
                    i = 0
                if i > nx-1:
                    i = nx-1
                ii = np.array([int(np.floor(i)), int(np.ceil(i))])
            else:
                i = i%nx
                ii = np.array([int(np.floor(i)), int(np.ceil(i))])
                if i > nx-1:
                    ii = np.array([int(np.floor(i)), 0])

            j = (xx[1]-Oy)/dy
            jj = np.array([int(np.floor(j)), int(np.ceil(j))])
            if not self.periodic[1]:
                if j < 0:
                    j = 0
                if j > ny-1:
                    j = ny-1
                jj = np.array([int(np.floor(j)), int(np.ceil(j))])
            else:
                j = j%ny
                jj = np.array([int(np.floor(j)), int(np.ceil(j))])
                if j > ny-1:
                    jj = np.array([int(np.floor(j)), 0])

            k = (xx[2]-Oz)/dz
            kk = np.array([int(np.floor(k)), int(np.ceil(k))])
            if not self.periodic[2]:
                if k < 0:
                    k = 0
                if k > nz-1:
                    k = nz-1
                kk = np.array([int(np.floor(k)), int(np.ceil(k))])
            else:
                k = k%nz
                kk = np.array([int(np.floor(k)), int(np.ceil(k))])
                if k > nz-1:
                    kk = np.array([int(np.floor(k)), 0])

        # Interpolate the field.
        if self.interpolation == 'mean':
            sub_field = [self.u[[ii[0], ii[1]], [jj[0], jj[1]], [kk[0], kk[1]]],
                         self.v[[ii[0], ii[1]], [jj[0], jj[1]], [kk[0], kk[1]]],
                         self.w[[ii[0], ii[1]], [jj[0], jj[1]], [kk[0], kk[1]]]]
            return np.mean(np.array(sub_field), axis=(1, 2, 3))

        if self.interpolation == 'trilinear':
            if ii[0] == ii[1]:
                w1 = np.array([1, 1])
            else:
                if (i > nx-1) and (self.periodic[0]):
                    w1 = np.array([nx-i, i-ii[0]])
                else:
                    w1 = (i-ii[::-1])

            if jj[0] == jj[1]:
                w2 = np.array([1, 1])
            else:
                if (j > ny-1) and (self.periodic[1]):
                    w2 = np.array([ny-j, j-jj[0]])
                else:
                    w2 = (j-jj[::-1])

            if kk[0] == kk[1]:
                w3 = np.array([1, 1])
            else:
                if (k > nz-1) and (self.periodic[2]):
                    w3 = np.array([nz-k, k-kk[0]])
                else:
                    w3 = (k-kk[::-1])

            weight = abs(w1.reshape((2, 1, 1))*w2.reshape((1, 2, 1))*w3.reshape((1, 1, 2)))
            sub_field = [self.u[[ii[0], ii[1]]][:, [jj[0], jj[1]]][:, :, [kk[0], kk[1]]],
                         self.v[[ii[0], ii[1]]][:, [jj[0], jj[1]]][:, :, [kk[0], kk[1]]],
                         self.w[[ii[0], ii[1]]][:, [jj[0], jj[1]]][:, :, [kk[0], kk[1]]]]
            return np.sum(np.array(sub_field)*weight, axis=(1, 2, 3))/np.sum(weight)

        # If the point lies outside the domain, return 0.
        if (ii[0] < -1) or (ii[1] > nx) or (jj[0] < -1) or (jj[1] > ny) \
            or (kk[0] < -1) or (kk[1] > nz):
            return np.zeros([0, 0, 0])


    def __set_material(self, idx, color_rgba):
        """
        Set the mesh material.

        call signature:

        __set_material(idx, color_rgba):

        Keyword arguments:

        *idx*:
          Index of the material.

        *color_rgba*:
          The rgba values of the colors to be used.
        """

        import bpy
        import numpy as np

        # Deterimne if we need a list of materials, i.e. for every arrow mesh one.
        if any([isinstance(self.color, np.ndarray),
                isinstance(self.emission, np.ndarray),
                isinstance(self.roughness, np.ndarray)]):
            list_material = True
        else:
            list_material = False

        # Transform single values to arrays.
        if list_material:
            if color_rgba.shape[0] != self.seeds.shape[0]:
                color_rgba = np.repeat(color_rgba, self.seeds.shape[0], axis=0)
            if not isinstance(self.roughness, np.ndarray):
                self.roughness = np.ones(self.seeds.shape[0])*self.roughness
            if not self.emission is None:
                if not isinstance(self.emission, np.ndarray):
                    self.emission = np.ones(self.seeds.shape[0])*self.emission

        # Set the material.
        if list_material:
            self.mesh_material.append(bpy.data.materials.new('material'))
            self.curve_object[idx].active_material.active_material = self.mesh_material[idx]
        else:
            if idx == 0:
                self.mesh_material.append(bpy.data.materials.new('material'))
                self.mesh_material[0].diffuse_color = color_rgba[idx]
            self.curve_object[idx].active_material = self.mesh_material[0]

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
                node_emission.inputs['Color'].default_value = tuple(color_rgba[idx]) + (1, )
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
                node_emission.inputs['Color'].default_value = color_rgba[idx] + (1, )
                if isinstance(self.emission, np.ndarray):
                    node_emission.inputs['Strength'].default_value = self.emission
                else:
                    node_emission.inputs['Strength'].default_value = self.emission
