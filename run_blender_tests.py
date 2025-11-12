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

# Load your test file(s).
TEST_FILE = Path(__file__).parent / "tests/test_all.py"
spec = importlib.util.spec_from_file_location("blendaviz_tests", TEST_FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
result = unittest.TextTestRunner(verbosity=2).run(suite)

# Stop coverage and save reports.
cov.stop()
cov.save()
cov.xml_report(outfile="coverage.xml")
cov.html_report(directory="coverage_html")

# Exit Blender cleanly.
sys.stdout.flush()
sys.stderr.flush()
bpy.ops.wm.quit_blender()







