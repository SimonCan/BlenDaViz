#
# The __init__ file is used not only to import the sub-modules, but also to
# set everything up properly.
#

import bpy

# Load sub-modules.
from .generic import GenericPlot
from .plot1d import plot, PathLine
from .plot2d import mesh, Surface
from .plot3d import vol, Volume, quiver, Quiver3d, contour, Contour3d
from .streamlines3d import streamlines_function, streamlines_array, \
    Streamline3d, Streamline3dArray
from .matplotlib_bridge import mpl_figure_to_blender, MPLEmbedding
from .colors import string_to_rgba, make_rgba_array
from .vectors import vec, arrow
from .globals import HouseKeeping
from .box import bounding_box, BoundingBox
from .lights import adjust_lights
from .camera import adjust_camera


# Initialize our plot stack and global housekeeping object.
plot_stack = []
house_keeping = HouseKeeping()

# import inspect

# Remove plot from stack when deleting the geometry.
def delete_plot_object(obj):
    # curframe = inspect.currentframe()
    # calframe = inspect.getouterframes(curframe, 1)
    # print('caller name:', calframe[1])
    print("delete_plot_object")
    stack_remove_list = []
    # Find the plot objects that will be removed from the stack.
    for plot_obj in plot_stack:
        stack_obj = plot_obj.deletable_object
        if stack_obj == obj:
            stack_remove_list.append(plot_obj)
    for i, light in enumerate(house_keeping.lights):
        if light == obj:
            house_keeping.lights[i] = None
    if obj == house_keeping.camera:
        house_keeping.camera = None
    if obj == house_keeping.box:
        house_keeping.box = None
    # Remove all plot objects connected to the deleted geometry.
    for plot_obj in stack_remove_list:
        plot_stack.remove(plot_obj)

bpy.types.Object.__del__ = delete_plot_object

# Add the needed deletable_object object attribute to the Blender light class.
setattr(bpy.types.Object, 'deletable_object', None)
