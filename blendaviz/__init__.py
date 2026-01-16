#
# The __init__ file is used not only to import the sub-modules, but also to
# set everything up properly.
#

import bpy

# Load sub-modules.
from .generic import GenericPlot
from .plot1d import plot, PathLine
from .plot2d import mesh, Surface
from .plot3d import quiver, Quiver3d, contour, Contour3d
from .streamlines3d import streamlines_function, streamlines_array, \
      Streamline3d, Streamline3dArray
from .matplotlib_bridge import mpl_figure_to_blender, MPLEmbedding
from .colors import string_to_rgba, make_rgba_array
from .globals import HouseKeeping
from .box import bounding_box, BoundingBox
from .lights import adjust_lights
from .camera import adjust_camera

# Load utility modules (for internal use and advanced users)
from . import markers
from . import materials


# Initialize our plot stack and global housekeeping object.
plot_stack = []
house_keeping = HouseKeeping()

# Add the needed deletable_object object attribute to the Blender light class.
setattr(bpy.types.Object, 'deletable_object', None)
