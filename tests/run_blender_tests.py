"""
Launcher to run all unittests inside Blender properly.
Run it like:
~/bin/blender-4.4.3-linux-x64/blender --background --factory-startup --python run_blender_tests.py
"""

import unittest
import bpy
import sys
import importlib.util
from pathlib import Path

# Path to your test file(s)
TEST_FILE = Path(__file__).parent / "test_all.py"

# Dynamically import the test module so it has a proper __name__
spec = importlib.util.spec_from_file_location("blendaviz_tests", TEST_FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Now we can load the test classes from that module
suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
result = unittest.TextTestRunner(verbosity=2).run(suite)

sys.stdout.flush()
sys.stderr.flush()
bpy.ops.wm.quit_blender()
