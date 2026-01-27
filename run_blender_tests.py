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

# Ensure required packages are available inside Blender's Python environment.
def ensure_packages(packages):
    """Install packages if not already available."""
    missing = []
    for package_name, import_name in packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "ensurepip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

# Install all required dependencies.
ensure_packages([
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("scikit-image", "skimage"),
    ("coverage", "coverage"),
])

import coverage

# Ensure project root is on sys.path (so "import blendaviz" works).
ROOT = Path(__file__).resolve().parent
if (ROOT / "blendaviz").exists():
    sys.path.insert(0, str(ROOT))
elif (ROOT.parent / "blendaviz").exists():
    sys.path.insert(0, str(ROOT.parent))

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







