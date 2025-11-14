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
        y = np.linspace(0, 6*np.pi, 20)
        x = 2*np.cos(y/2)
        z = 2*np.sin(y/2)

        # Line plot.
        pl_1 = blt.plot(x, y, z, radius=0.1, color='red')
        # Marker plot.
        pl_2 = blt.plot(x, y, z, marker='cube', radius=0.7, color='b')

        self.assertIsNotNone(pl_1, "blt.plot() returned None")
        self.assertIsNotNone(pl_2, "blt.plot() returned None")

        objects = list(bpy.data.objects)
        self.assertGreater(len(objects), 0, "No Blender objects were created by plot().")


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Plot1d)  # ✅ fixed class name
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.stdout.flush()
    sys.stderr.flush()
    bpy.ops.wm.quit_blender()

    return result


run_tests()
