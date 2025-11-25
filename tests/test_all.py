"""
Run all BlenDaViz tests inside Blender:
"""

import unittest
import numpy as np
import blendaviz as blt
import bpy
import sys


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
        cube = bpy.ops.mesh.primitive_cube_add()
        pl = blt.plot(x, y, z, radius=0.1, color='red', marker=bpy.context.object)
        self.assertIsNotNone(pl, "blt.plot() returned None")

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

        # Make sure no additional Blender objects were created.
        objects = list(bpy.data.objects)
        self.assertGreater(len(objects), 0, "No Blender objects were created by plot().")


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(Plot1d))
    suite.addTests(loader.loadTestsFromTestCase(Plot2d))

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.stdout.flush()
    sys.stderr.flush()
    bpy.ops.wm.quit_blender()
    return result

run_tests()
