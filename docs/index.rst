.. BlenDaViz documentation master file, created by
   sphinx-quickstart on Fri Feb 26 14:34:58 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***************
BlenDaViz
***************

Set Up
======

BlenDaViz requires Blender version 2.8 or above and is not compatible with any previous version due to syntax changes. It also requires the Python libraries numpy, scipy, eqtools and matplotlib, which can be installed via your package manager or via pip.

Note that the version of the library must correspond to the version of the inbuilt Python of Blender, rather than the one installed on your system. If both match you are lucky and have an easy installation. If not you need to follow some simple steps, as explained below.

After the successful installation of Blender and the required libraries you need to download the BlenDaViz library and can store it wherever on your PC, say :code:`~/libs/blendaviz`. You now need to set your :code:`PYTHONPATH` variable on your system, which can be different depending on what operating system you are using and what shell (bash or tcsh). On any Unix system (Linux, MacOS, BSD) using bash just add this to your :code:`~/.bashrc`:
.. code:: bash

   export PYTHONPATH=$PYTHONPATH:~/libs


To make sure Python in Blender uses this environmental variable, you need to start Blender using:
.. code:: bash

   blender --python-use-system-env


Once you start Blender, open a Python console in Blender and type :code:`import blendaviz as blt` and you are good to go.


Install Blender 2.80+ and Libraries Locally
-------------------------------------------

If your already installed Blender and Python libraries work together, there is no need to read this section.
If you are trying to use BlenDaViz on a computer that requires a manual installation of Blender and the libraries or if you are on a computer for which you do not have root/admin access follow this guide.

1. **Download and extract/install [Blender](https://www.blender.org/download/) 2.80+.**

2. **Extract the file.**
There will be an executable within the directory.
Let us call this directory :code:`blender_dir`.

3. **Install the required Python libraries for the correct version.** </br>
For this you need to verify the version of the inbuilt Python of Blender.
You find this in the directory :code:`4.1/python/bin/` and replace :code:`4.1` with the right Blender version.
For Blender 4.1.1 this is :code:`python3.11`.
Now install the required libraries using pip for the correct Python version:
.. code:: bash

   cd blender_dir/2.8X/python/bin
   ./python3.11 -m ensurepip
   ./python3.11 -m pip install numpy
   ./python3.11 -m pip install scipy
   ./python3.11 -m pip install eqtools
   ./python3.11 -m pip install matplotlib
   ./python3.11 -m pip install scikit-image

4. **Upgrade the installed libraries (recommended).**
You should do this step in case you are getting error messages when using the plotting commands.
.. code:: bash
   ./python3.11 -m pip install numpy --upgrade
   ./python3.11 -m pip install scipy --upgrade
   ./python3.11 -m pip install eqtools --upgrade
   ./python3.11 -m pip install matplotlib --upgrade



   Usage and Examples
==================

Open Blender and within Blender a Python console.
Import BlendaViz and numpy:
.. code:: python
   import blendaviz as blt
   import numpy as np


Marker Plots
------------

In this simple line plot we will create some data and plot them. We will also see how we can manipulate existing plots.
.. code:: python
   # Create the data.
   y = np.linspace(0, 6*np.pi, 20)
   x = 2*np.cos(y/2)
   z = 2*np.sin(y/2)
   # Generate the scatter plot.
   pl = blt.plot(x, y, z, marker='cube', radius=0.7)
   # Change the color.
   pl.color = np.ones([x.shape[0], 4])
   pl.color[:, 0]  = np.linspace(0, 1, 20)
   pl.color[:, 1] = 0
   pl.plot()

Now you can render the scene by pressing F12.

.. ![MarkerPlot](https://github.com/SimonCan/BlenDaViz/blob/master/docs/marker_plot.png)


Line Plots
----------

A line plot is very similar to a marker plot. It draws the data points as a line/tube.

.. code:: python
   import blendaviz as blt
   import numpy as np
   # Generate the data.
   y = np.linspace(0, 6*np.pi, 400)
   x = 2*np.cos(y)
   z = 2*np.sin(y)
   # Generate the line plot.
   pl = blt.plot(x, y, z, radius=0.5)

.. ![LinePlot](https://github.com/SimonCan/BlenDaViz/blob/master/docs/line_plot.png)


Mesh Plots
----------
We can plot 2d data arrays using :code:`mesh`. We need two 2d arrays containing the x and y coordinates of the data points.
.. code:: python
   import numpy as np
   import blendaviz as blt
   # Generate the data.
   x0 = np.linspace(-3, 3, 20)
   y0 = np.linspace(-3, 3, 20)
   x, y = np.meshgrid(x0, y0, indexing='ij')
   z = (1 - x**2-y**2)*np.exp(-(x**2+y**2)/5)
   # Genereate the mesh plot.
   mesh = blt.mesh(x, y, z)

.. ![MeshPlot](https://github.com/SimonCan/BlenDaViz/blob/master/docs/mesh_plot.png)


.. ## Quiver Plots
.. For three-dimensional vector arrays we can user quiver to plot the vector field as arrows. We need the x, y and z-coordinates of the data points as 3d arrays.
.. ```python
.. import numpy as np
.. import blendaviz as blt
.. # Generate the data.
.. x = np.linspace(-3, 3, 3)
.. y = np.linspace(-7, 7, 7)
.. z = np.linspace(-3, 3, 3)
.. xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
.. uu = 0.3*(xx + yy)
.. vv = 0.3*xx
.. ww = 0.3*zz + 0.8
.. # Genereate the quiver plot.
.. quiver = blt.quiver(xx, yy, zz, uu, vv, ww, length='magnitude', color='magnitude')
.. ```
..
.. ![QuiverPlot](https://github.com/SimonCan/BlenDaViz/blob/master/docs/quiver_plot.png)
..
..
.. ## Contour Plots
.. Three-dimensional scalar fields can be plotted using `contour`. We need the x, z and z-coordinates of the data points as 3d arrays.
..
.. ```python
.. import blendaviz as blt
.. # Generate the data.
.. x = np.linspace(-2, 2, 21)
.. y = np.linspace(-2, 2, 21)
.. z = np.linspace(-2, 2, 21)
.. xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
.. phi = np.sin(3*xx) + np.cos(2*yy) + np.sin(zz)
.. # Genereate the contour plot.
.. contour = blt.contour(phi, xx, yy, zz)
.. ```
..
.. ![ContourPlot](https://github.com/SimonCan/BlenDaViz/blob/master/docs/contour_plot.png)
..
..
.. ## Streamline Plots
.. A three-dimensional vector field can be plotted as streamlines. For that we need specify the three components of the vector field as 3d arrays, the coordinates of the data points as 3d arrays and the position or number of seeds. If the number of seeds is passed, they will be randomly distributed within the domain.
.. ```python
.. import numpy as np
.. import blendaviz as blt
.. # Generate the data.
.. x = np.linspace(-4, 4, 100)
.. y = np.linspace(-4, 4, 100)
.. z = np.linspace(-4, 4, 100)
.. xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
.. u = -yy*np.exp(-np.sqrt(xx**2+yy**2) - zz**2)
.. v = np.ones_like(u)*0.1
.. w = xx*np.exp(-np.sqrt(xx**2+yy**2) - zz**2)
.. # Define the position of the seeds.
.. seeds = np.array([np.random.random(10)*2-1, np.zeros(10), np.random.random(10)*2-1]).T
.. # Generate the streamline plot.
.. streamlines = blt.streamlines(x, y, z, u, v, w, seeds=seeds, integration_time=100, integration_steps=80)
.. ```
..
.. ![StreamlinePlot](https://github.com/SimonCan/BlenDaViz/blob/master/docs/streamlines_plot.png)
..
..
.. # Plotting Without the Blender GUI
.. There are a few occasions that you do not want to start the Blender GUI, like you don't have any compatible graphics driver installed on your system, you want to run the plot in the background, or you are on a cluster with SSH access. Computationally intensive rendering should be ideally done on a powerful computer and done over night or even several days. Here we outline how to make a plot without the GUI.
..
.. We present two methods for performing a background plot. The first method is the easier one, but it requires the usage of the GUI to set up the scene, i.e. light, camera and additional rendering options. The second methods requires more coding, but is runs purely in the background.
..
.. ### Using the GUI to set up the scene.
..
.. 1. Open Blender and remove the default cube.
.. 2. Adjust any other scene and rendering options.
.. 3. Save your scence to something like ```my_plot.blend```.
.. 4. Prepare the plotting routine using BlenDaViz and save it to something like ```my_plot.py```.
.. 5. Start Blender from the command line using the prepared scene and the plotting script.
.. ```bash
.. blender --background my_plot.blend -P my_plot.py
.. ```
.. This will use your blender scene and execute the plotting script.
..
.. ### Preparing the scene without the GUI, using the Blender Python commands.
.. This requires a few lines of coding, as we perform all of the steps done in the GUI using the Blender Python commands. Not all of the below steps are required, but highly recommended.
..
.. The steps in the script are basically:
.. 1. Remove any existing objects from the default scene, like the default cube at the origin.
.. 2. (Optionally, Recommended) Adjust the background and rendering options.
.. 3. Perform the BlenDaViz plot.
.. 4. Render the scene and save the image.
..
.. You then need to run the script using
.. ```bash
.. blender -P my_script.py
.. ```
.. It should be evident that using a loop you can generate animations through a sequence of images. You can use ffmpeg to put the images into a video file.
..
.. ```python
.. # line_plot_background.py
.. '''
.. Plotting example for a line plot in the background.
.. Usage:
.. blender -P line_plot_background.py
.. '''
..
.. import blendaviz as blt
.. import numpy as np
.. import bpy
..
.. # Delete all existing objects, like the default cube, light and camera.
.. bpy.ops.object.select_all(action='SELECT')
.. bpy.ops.object.delete(use_global=False)
..
.. # Change the background color.
.. bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
.. bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 5
.. bpy.context.scene.world.cycles_visibility.scatter = False
.. bpy.context.scene.world.cycles_visibility.diffuse = False
.. bpy.context.scene.world.cycles_visibility.glossy = False
.. bpy.context.scene.world.cycles_visibility.transmission = False
..
.. # Change the rendering options.
.. bpy.context.scene.render.engine = 'CYCLES'
.. bpy.context.scene.render.threads_mode = 'FIXED'
.. bpy.context.scene.render.threads = 4
.. bpy.context.scene.cycles.samples = 256
.. bpy.context.scene.render.resolution_x = 1920
.. bpy.context.scene.render.resolution_y = 1080
..
.. # Generate the data.
.. y = np.linspace(0, 6*np.pi, 400)
.. x = 2*np.cos(y)
.. z = 2*np.sin(y)
..
.. # Generate the line plot.
.. pl = blt.plot(x, y, z, radius=0.5)
..
.. # Render the image.
.. bpy.data.scenes['Scene'].render.filepath = 'line_plot.png'
.. bpy.ops.render.render(write_still=True)
.. ```


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
