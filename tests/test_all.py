"""
Run all BlenDaViz tests inside Blender:
"""

import unittest
import numpy as np
import blendaviz as blt
import bpy
import matplotlib.pyplot as plt


class Plot1d(unittest.TestCase):
    def test_plot1d(self):
        # Define the numbr of data points.
        n = 20

        # Generate the data.
        y = np.linspace(-3*np.pi, 3*np.pi, n)
        x = 2*np.cos(y/2)
        z = 2*np.sin(y/2)

        # Line plot.
        pl = blt.plot(x, y, z, radius=0.1, color='red')
        self.assertIsNotNone(pl, "blt.plot() returned None")

        # Line plot with variable color.
        pl.color = np.random.random([n, 3])
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")
        pl.color = np.random.random([n, 4])
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")

        # Line plot with emission.
        pl.color = 'r'
        pl.emission = 10
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")
        pl.marker = 'cube'
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")
        pl.emission = 10 + np.linspace(0, 10, n)
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")
        pl.emission = None

        # Marker plot.
        pl.marker = 'cube'
        pl.radius = 0.7
        pl.color = 'b'
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")

        # Test different markers and change in range.
        pl.x = 2*pl.x
        pl.y = 2*pl.y
        pl.z = 2*pl.z
        pl.marker = 'cone'
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")
        pl.marker = 'cylinder'
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")
        pl.marker = 'ico_sphere'
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")
        pl.marker = 'monkey'
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")
        pl.marker = 'torus'
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")
        pl.marker = 'uv_sphere'
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() returned None")

        # Test custom marker.
        bpy.ops.mesh.primitive_cube_add()
        pl = blt.plot(x, y, z, radius=0.1, color='red',
                      marker=bpy.context.object)
        self.assertIsNotNone(pl, "blt.plot() returned None")

        # Test time-dependent data.
        n_time = 5
        x_t = np.zeros([n, n_time])
        y_t = np.zeros([n, n_time])
        z_t = np.zeros([n, n_time])
        for t_idx in range(n_time):
            phase = t_idx * np.pi / n_time
            y_t[:, t_idx] = np.linspace(-3*np.pi, 3*np.pi, n)
            x_t[:, t_idx] = 2*np.cos(y_t[:, t_idx]/2 + phase)
            z_t[:, t_idx] = 2*np.sin(y_t[:, t_idx]/2 + phase)
        time = np.linspace(0, n_time-1, n_time)
        pl = blt.plot(x_t, y_t, z_t, radius=0.1, color='green', time=time)
        self.assertIsNotNone(pl, "blt.plot() with time returned None")
        # Simulate frame change to test time_handler.
        bpy.context.scene.frame_set(2)
        pl.plot()
        self.assertIsNotNone(pl, "blt.plot() after frame change returned None")

        # Make sure no additional Blender objects were created.
        objects = list(bpy.data.objects)
        self.assertGreater(len(objects), 0, "No Blender objects were created by plot().")


class Plot2d(unittest.TestCase):
    def test_plot2d(self):
        # Generate the data.
        x0 = np.linspace(-3, 3, 51)
        y0 = np.linspace(-3, 3, 51)
        x, y = np.meshgrid(x0, y0, indexing='ij')
        z = (1 - x**2-y**2)*np.exp(-(x**2+y**2)/5)

        # Generate the mesh plot.
        mesh = blt.mesh(x, y, z)
        self.assertIsNotNone(mesh, "blt.plot() returned None")

        # Change the colors.
        mesh.c = 'red'
        mesh.plot()
        self.assertIsNotNone(mesh, "blt.plot() returned None")
        mesh.c = x
        mesh.plot()
        self.assertIsNotNone(mesh, "blt.plot() returned None")
        mesh.alpha = x/(x.max() - x.min())
        mesh.plot()
        self.assertIsNotNone(mesh, "blt.plot() returned None")

        # Change the range to test global updates.
        mesh.x = 2*x
        mesh.y = 2*y
        mesh.z = z - 1
        mesh.plot()
        self.assertIsNotNone(mesh, "blt.plot() returned None")

        # Test time-dependent data.
        n_time = 5
        nx, ny = x.shape
        x_t = np.zeros([nx, ny, n_time])
        y_t = np.zeros([nx, ny, n_time])
        z_t = np.zeros([nx, ny, n_time])
        c_t = np.zeros([nx, ny, n_time])
        for t_idx in range(n_time):
            phase = t_idx * np.pi / n_time
            x_t[:, :, t_idx] = x
            y_t[:, :, t_idx] = y
            z_t[:, :, t_idx] = (1 - x**2 - y**2) * np.exp(-(x**2 + y**2)/5) * np.cos(phase)
            c_t[:, :, t_idx] = x * np.sin(phase)
        time = np.linspace(0, n_time-1, n_time)
        mesh = blt.mesh(x_t, y_t, z_t, time=time)
        self.assertIsNotNone(mesh, "blt.mesh() with time returned None")
        # Test with time-dependent color.
        mesh.c = c_t
        mesh.plot()
        self.assertIsNotNone(mesh, "blt.mesh() with time-dependent color returned None")
        # Simulate frame change to test time_handler.
        bpy.context.scene.frame_set(2)
        mesh.plot()
        self.assertIsNotNone(mesh, "blt.mesh() after frame change returned None")
        # Test with time-dependent vmin/vmax.
        mesh.vmin = np.linspace(-1, -0.5, n_time)
        mesh.vmax = np.linspace(0.5, 1, n_time)
        mesh.plot()
        self.assertIsNotNone(mesh, "blt.mesh() with time-dependent vmin/vmax returned None")

        # Make sure no additional Blender objects were created.
        objects = list(bpy.data.objects)
        self.assertGreater(len(objects), 0, "No Blender objects were created by plot().")


class Plot3d(unittest.TestCase):
    def test_plot3d(self):
        # Generate the data.
        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 5)
        z = np.linspace(-2, 2, 5)
        u = np.array([1, 0, 0, 1, 0])
        v = np.array([0, 1, 0, 1, 1])
        w = np.array([0, 0, 1, 0, 1])

        # Generate the quiver plot.
        qu = blt.quiver(x, y, z, u, v, w, pivot='mid', color='red')
        self.assertIsNotNone(qu, "blt.plot() returned None")

        # Change length, pivot and color.
        qu.length = 'magnitude'
        qu.color = 'magnitude'
        qu.pivot = 'tip'
        qu.plot()
        self.assertIsNotNone(qu, "blt.plot() returned None")

        # Change pivot and color to fixed for all arrows.
        qu.pivot = 'tail'
        qu.color = 'blue'
        qu.plot()
        self.assertIsNotNone(qu, "blt.plot() returned None")

        # Test color arrays and lists.
        qu.color = np.random.random([5, 4])
        qu.plot()
        self.assertIsNotNone(qu, "blt.plot() returned None")
        qu.color = ['r', 'blue', (1, 1, 0, 1), 'k', 'green']
        qu.plot()
        self.assertIsNotNone(qu, "blt.plot() returned None")

        # Test emission.
        qu.emission = x / (x.max() - x.min())
        qu.plot()
        self.assertIsNotNone(qu, "blt.plot() returned None")
        qu.emission = 10.0
        qu.plot()
        self.assertIsNotNone(qu, "blt.plot() returned None")
        qu.color = 'r'
        qu.roughness = 1
        qu.emission = 10.0
        qu.plot()
        self.assertIsNotNone(qu, "blt.plot() returned None")

        # Contour plots.
        x = np.linspace(-2, 2, 21)
        y = np.linspace(-2, 2, 21)
        z = np.linspace(-2, 2, 21)
        xx, yy, zz = np.meshgrid(x, y, z)
        phi = xx**2 + yy**2 + zz**2
        iso = blt.contour(phi, xx, yy, zz, contours=[0.3, 0.6],
                          color=np.array([(1, 0, 0, 1), (0, 1, 0, 0.5)]))
        self.assertIsNotNone(iso, "blt.plot() returned None")

        # Test different contours.
        iso.contours = 4
        iso.roughness = 1.0
        iso.plot()
        self.assertIsNotNone(iso, "blt.plot() returned None")

        # Test emission.
        iso.emission = 10
        iso.plot()
        self.assertIsNotNone(iso, "blt.plot() returned None")
        iso.color = 'red'
        iso.plot()
        self.assertIsNotNone(iso, "blt.plot() returned None")

        # Test emission with single contour (list_material = False path).
        iso.contours = [0.5]
        iso.color = 'blue'
        iso.emission = 5.0
        iso.roughness = 0.5
        iso.plot()
        self.assertIsNotNone(iso, "blt.plot() returned None")

        # Test coloring using a different scalar field.
        iso.psi = xx + yy
        iso.plot()
        self.assertIsNotNone(iso, "blt.plot() returned None")

        # Test time-dependent quiver data.
        n_time = 3
        n_pts = 5
        x_q0 = np.linspace(-2, 2, n_pts)
        y_q0 = np.linspace(-2, 2, n_pts)
        z_q0 = np.linspace(-2, 2, n_pts)
        x_qt = np.zeros([n_pts, n_time])
        y_qt = np.zeros([n_pts, n_time])
        z_qt = np.zeros([n_pts, n_time])
        u_t = np.zeros([n_pts, n_time])
        v_t = np.zeros([n_pts, n_time])
        w_t = np.zeros([n_pts, n_time])
        for t_idx in range(n_time):
            phase = t_idx * 2 * np.pi / n_time
            x_qt[:, t_idx] = x_q0
            y_qt[:, t_idx] = y_q0
            z_qt[:, t_idx] = z_q0
            u_t[:, t_idx] = np.cos(phase)
            v_t[:, t_idx] = np.sin(phase)
            w_t[:, t_idx] = 0.5
        time = np.linspace(0, n_time-1, n_time)
        qu = blt.quiver(x_qt, y_qt, z_qt, u_t, v_t, w_t, time=time, color='cyan')
        self.assertIsNotNone(qu, "blt.quiver() with time returned None")
        # Simulate frame change.
        bpy.context.scene.frame_set(1)
        qu.plot()
        self.assertIsNotNone(qu, "blt.quiver() after frame change returned None")

        # Test time-dependent contour data.
        nx, ny, nz = 11, 11, 11
        x_c = np.linspace(-2, 2, nx)
        y_c = np.linspace(-2, 2, ny)
        z_c = np.linspace(-2, 2, nz)
        xx_c, yy_c, zz_c = np.meshgrid(x_c, y_c, z_c, indexing='ij')
        phi_t = np.zeros([nx, ny, nz, n_time])
        for t_idx in range(n_time):
            scale = 1 + 0.5 * t_idx
            phi_t[:, :, :, t_idx] = xx_c**2 + yy_c**2 + zz_c**2 / scale
        # Note: contour expects 1D x, y, z arrays, not meshgrid
        iso = blt.contour(phi_t, x_c, y_c, z_c, contours=[1.0], time=time, color='yellow')
        self.assertIsNotNone(iso, "blt.contour() with time returned None")
        # Simulate frame change.
        bpy.context.scene.frame_set(2)
        iso.plot()
        self.assertIsNotNone(iso, "blt.contour() after frame change returned None")

        # Make sure no additional Blender objects were created.
        objects = list(bpy.data.objects)
        self.assertGreater(len(objects), 0, "No Blender objects were created by plot().")


class Streamlines3d(unittest.TestCase):
    def test_streamlines3d(self):
        # Generate the data.
        def irrational_hopf(t, x):
            return 1/(1+np.sum(x[0]**2+x[1]**2+x[2]**2))**3 * \
                   np.array([2*(np.sqrt(2)*x[1] - x[0]*x[2]),
                            -2*(np.sqrt(2)*x[0] + x[1]*x[2]),
                            (-1 + x[0]**2 + x[1]**2 - x[2]**2)])

        # Generate a streamline plot.
        stream = blt.streamlines_function(irrational_hopf, n_seeds=5,
                                          integration_time=1000,
                                          integration_steps=500)
        self.assertIsNotNone(stream, "blt.plot() returned None")

        # Test emissions.
        stream.emission = 10
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")
        stream.n_seeds = 10
        stream.emission = np.linspace(10, 20, 10)
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")

        # Test textured streamlines for function-based streamlines.
        stream.emission = None
        stream.color_scalar = 'magnitude'
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")
        # Test with a custom scalar function.
        stream.color_scalar = lambda x: np.array([x[0], x[1], x[2]])
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")

        # Streamline plot using data array.
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        z = np.linspace(-4, 4, 100)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        u = -yy*np.exp(-np.sqrt(xx**2+yy**2) - zz**2)
        v = xx*np.exp(-np.sqrt(xx**2+yy**2) - zz**2)
        w = np.ones_like(u)*0.1
        stream = blt.streamlines_array(x, y, z, u, v, w,
                                       n_seeds=2, integration_time=2,
                                       color_scalar='magnitude', seed_radius=3)
        self.assertIsNotNone(stream, "blt.plot() returned None")

        # Test what happens near the boundary.
        u = xx
        v = yy
        w = zz
        stream.interpolation = 'trilinear'
        stream.seeds = np.array([[3.9, 3.9, 3.9], [-3.9, -3.9, -3.9]])
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")
        stream.periodic = (True, True, True)
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")
        u = -xx
        v = -yy
        w = -zz
        stream.interpolation = 'mean'
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")
        stream.periodic = (False, False, False)
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")

        # Test textured streamlines.
        stream.color_scalar = 'magnitude'
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")
        stream.color_scalar = stream.u
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")

        # Test multiprocessing.
        stream.n_proc = 2
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")

        # Test tricubic interpolation with mocked eqtools.
        import sys
        from unittest.mock import MagicMock

        # Create a mock Spline class that mimics eqtools.trispline.Spline
        class MockSpline:
            def __init__(self, z, y, x, data):
                from scipy.interpolate import RegularGridInterpolator
                # Swap axes back to match expected order
                self._interp = RegularGridInterpolator((x, y, z), np.swapaxes(data, 0, 2))
            def ev(self, z, y, x):
                # Return shape (1,) to match expected eqtools.trispline.Spline behavior
                return np.array([self._interp((x, y, z))])

        mock_eqtools = MagicMock()
        mock_eqtools.trispline.Spline = MockSpline
        sys.modules['eqtools'] = mock_eqtools
        sys.modules['eqtools.trispline'] = mock_eqtools.trispline

        stream.interpolation = 'tricubic'
        stream.n_proc = 1
        stream.plot()
        self.assertIsNotNone(stream, "blt.plot() returned None")

        # Clean up mock
        del sys.modules['eqtools']
        del sys.modules['eqtools.trispline']

        # Test time-dependent streamlines.
        n_time = 3
        n_grid = 20
        x_s = np.linspace(-4, 4, n_grid)
        y_s = np.linspace(-4, 4, n_grid)
        z_s = np.linspace(-4, 4, n_grid)
        xx_s, yy_s, zz_s = np.meshgrid(x_s, y_s, z_s, indexing='ij')
        u_t = np.zeros([n_grid, n_grid, n_grid, n_time])
        v_t = np.zeros([n_grid, n_grid, n_grid, n_time])
        w_t = np.zeros([n_grid, n_grid, n_grid, n_time])
        for t_idx in range(n_time):
            phase = t_idx * np.pi / n_time
            u_t[:, :, :, t_idx] = -yy_s * np.cos(phase)
            v_t[:, :, :, t_idx] = xx_s * np.cos(phase)
            w_t[:, :, :, t_idx] = 0.1 * np.sin(phase)
        time = np.linspace(0, n_time-1, n_time)
        stream = blt.streamlines_array(x_s, y_s, z_s, u_t, v_t, w_t,
                                       n_seeds=2, integration_time=1,
                                       time=time, seed_radius=2)
        self.assertIsNotNone(stream, "blt.streamlines_array() with time returned None")
        # Simulate frame change.
        bpy.context.scene.frame_set(1)
        stream.plot()
        self.assertIsNotNone(stream, "blt.streamlines_array() after frame change returned None")

        # Make sure no additional Blender objects were created.
        objects = list(bpy.data.objects)
        self.assertGreater(len(objects), 0, "No Blender objects were created by plot().")


class MPLEmbedding(unittest.TestCase):
    def test_mpl(self):
        # Create some data.
        x = np.linspace(0, 5, 1000)

        # Create the Matplotlib plot.
        fig = plt.figure()
        plt.plot(x, np.sin(x), color='g')
        plt.title("test")

        # Plot in Blender.
        mpl = blt.mpl_figure_to_blender(fig)
        self.assertIsNotNone(mpl, "mpl.plot() returned None")

        # Test replotting routine.
        mpl.position = [-1, -1, -1]
        mpl.plot()
        self.assertIsNotNone(mpl, "mpl.plot() returned None")


class Materials(unittest.TestCase):
    def test_materials(self):
        from blendaviz import materials

        # Test create_textured_material with emission.
        image = bpy.data.images.new('TestImage', 16, 16)
        mat = materials.create_textured_material('test_emission_mat', image, emission=5.0)
        self.assertIsNotNone(mat, "create_textured_material() with emission returned None")
        bpy.data.materials.remove(mat)

        # Test create_textured_material without emission (principled BSDF path).
        mat = materials.create_textured_material('test_bsdf_mat', image, roughness=0.5)
        self.assertIsNotNone(mat, "create_textured_material() without emission returned None")
        bpy.data.materials.remove(mat)

        # Clean up test image.
        bpy.data.images.remove(image)
