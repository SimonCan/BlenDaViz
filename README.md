# BlenDaViz
Scientific data visualization in Blender.

BlenDaViz is a Python library for Blender made for scientific data visualization. It can be used directly in the Blender Python console.

![Mesh plot](doc/mesh_plot.png?raw=true "Mesh plot in BlenDaViz")
![[doc/mesh_plot.png](https://raw.githubusercontent.com/SimonCan/BlenDaViz/master/docs/mesh_plot.png)|alt=Mesh plot]]

![MarkerPlot](https://github.com/SimonCan/BlenDaViz/blob/master/docs/mesh_plot.png)

# Set Up (Short Version)
Clone BlenDaViz on your computer:
```
git clone git@github.com:SimonCan/BlenDaViz.git
```
Add the root directory of your BlenDaViz directory to your PYTHONPATH variable (here shown for bash):
```
export PYTHONPATH=$PYTHONPATH:~/root_dir_to_blendaviz
```
Start blender (version 2.80+):
```
blender
```
Open a console in Blender.

Import the library and make a simple plot from numpy data:
```
import blendaviz as blt
import numpy as np
z = np.linspace(0, 6*np.pi, 30)
x = 3*np.cos(z)
y = 3*np.sin(z)
pl = blt.plot(x, y, z, marker='cube', radius=0.5)
```

A more detailed set up guide, including the local installation of blender 2.8 and libraries, such as scipy, can be found in the wiki.
