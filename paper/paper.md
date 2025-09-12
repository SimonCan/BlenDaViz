---
title: 'BlenDaViz, scientific visualization library for Blender in Python'
tags:
  - Python
  - data visualization
  - scientific visualization
  - Blender
  - ray tracing
authors:
  - name: Simon Candelaresi
    orcid: 0000-0002-7666-8504
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Christopher Berg Smiet
    orcid: 0000-0002-7803-7685
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: High-Performance Scientific Computing, University of Augsburg, Universit\"atsstra\ss e 12a, Augsburg, 86159, Germany
   index: 1
 - name: Swiss Plasma Center, École Polytechnique Fédérale de Lausanne (EPFL), Station 13, Lausanne, CH-1015, Switzerland
   index: 2
date: 20 September 2025
bibliography: paper.bib
---

# Summary

BlenDaViz is a 3d plotting library for the visualization tool Blender, written in Python.
It is able to visualize line plots, scatter plots, surfaces,
iso-surfaces, glyphs (arrows) and streamlines from Numpy arrays.
The user can load BlenDaViz within a Blender Python console as any other library.
Its design is close to other plotting libraries, especially Matplotlib, which means that
every part of a plot is treated as an object that can be changed when updating the plot
with new parameters or data.
Compared to other 3d visualization software, we achieve higher image quality, thanks to Blender's
ray tracing capabilities.


# Statement of need

In scientific fields, like structure mechanics, computational fluid dynamics,
astrophysics, or molecular dynamics, it is common to use 3d visualization tools
like Paraview, Vapor, Mayavi or Visit.
They are capable to load a variety of data formats and visualize different
aspects of the data using filters.
However, they often lack the image quality needed that would make understanding
the data easy.
Their limited shaders can create optical illusions where the depth of the geometry
is not clear, or the colors are off due to unrealistic image generation.
BlenDaViz fills this gap by giving the users all the rendering and ray tracing
capabilities of Blender, combined with the ease of use and modularity we know
from libraries like Matplotlib.



# Mathematics


# Citations


# Figures


# Acknowledgements


# References
