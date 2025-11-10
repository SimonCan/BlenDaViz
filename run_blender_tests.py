"""
Run all unittests in Blender with coverage support.
Used in CI via: blender --background --factory-startup --python run_blender_tests.py
"""

import sys
import unittest
import importlib.util
import bpy
from pathlib import Path
import subprocess

# Ensure 'coverage' is available inside Blender's Python environment.
try:
    import coverage
except ImportError:
    print("Installing 'coverage' inside Blender's Python environment...")
    subprocess.check_call([sys.executable, "-m", "ensurepip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage"])
    import coverage

# Start coverage collection before imports.
cov = coverage.Coverage(source=["blendaviz"])
cov.start()

# Discover and import all tests in /tests.
test_dir = Path(__file__).parent / "tests"
suite = unittest.TestSuite()

# Loop over all tests.
for test_file in test_dir.glob("test_*.py"):
    spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(mod))

# Run the tests.
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# Stop and save coverage.
cov.stop()
cov.save()
cov.html_report(directory="coverage_html")
cov.xml_report(outfile="coverage.xml")

bpy.ops.wm.quit_blender()
sys.exit(0 if result.wasSuccessful() else 1)


