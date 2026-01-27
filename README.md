# BlenDaViz

Scientific data visualization in Blender.

BlenDaViz is a Python library for Blender made for scientific data visualization. It can be used directly in the Blender Python console.

![MeshPlot](https://github.com/SimonCan/BlenDaViz/blob/master/docs/mesh_plot.png)

## Requirements

- Blender 4.4 or above (including Blender 5.0)
- Python libraries: numpy, scipy, matplotlib, scikit-image

## Installation

1. Clone BlenDaViz:
```bash
git clone https://github.com/SimonCan/BlenDaViz.git
cd BlenDaViz
```

2. Install using Blender's Python:
```bash
/path/to/blenders/python -m pip install .
```

To find Blender's Python path, open Blender, switch to a Python console (Shift + F4) and type:
```python
import sys
print(sys.executable)
```

## Quick Start

Start Blender and open a Python console. Import the library and make a simple plot:

```python
import blendaviz as blt
import numpy as np

z = np.linspace(0, 6*np.pi, 30)
x = 3*np.cos(z)
y = 3*np.sin(z)

pl = blt.plot(x, y, z, marker='cube', radius=0.5)
```

Press F12 to render the scene.

## Documentation

A detailed guide with examples can be found in the [documentation](https://blendaviz.readthedocs.io/en/latest/).

## Citation

If you use BlenDaViz in your research, please cite it using the information in [CITATION.cff](CITATION.cff).

## License

BlenDaViz is released under the LGPL-3.0 license. See [LICENSE](LICENSE) for details.
