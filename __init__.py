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


# Initialize our global housekeeping object.
__stack__ = []
house_keeping = HouseKeeping()


# Override the object delete operation.
class DeleteOverride(bpy.types.Operator):
    """
    Override the mesh delete operation.
    """

    bl_idname = "object.delete"
    bl_label = "Object Delete Operator"

    @classmethod
    def poll(cls, context):
        try:
            active_object = context.active_object
        except:
            active_object = None
        return active_object is not None

    def execute(self, context):
        stack_remove_list = []
        # Find the plot objects that will be removed from the stack.
        for obj in context.selected_objects:
            for plot in __stack__:
                stack_obj = plot.deletable_object
                if stack_obj == obj:
                    stack_remove_list.append(plot)
            for i, light in enumerate(house_keeping.lights):
                if light == obj:
                    house_keeping.lights[i] = None
            if obj == house_keeping.camera:
                house_keeping.camera = None
            if obj == house_keeping.box:
                house_keeping.box = None
            bpy.data.objects.remove(obj)
        # Remove all plot objects connected to the deleted geometry.
        for plot in stack_remove_list:
            __stack__.remove(plot)
        return {'FINISHED'}

def register_delete_override():
    bpy.utils.register_class(DeleteOverride)

def unregister_delete_override():
    bpy.utils.unregister_class(DeleteOverride)

#register_delete_override()

# Remove plot from stack when deleting the geometry.
def delete_plot_object(obj):
    stack_remove_list = []
    # Find the plot objects that will be removed from the stack.
    for plot in __stack__:
        stack_obj = plot.deletable_object
        if stack_obj == obj:
            stack_remove_list.append(plot)
    for i, light in enumerate(house_keeping.lights):
        if light == obj:
            house_keeping.lights[i] = None
    if obj == house_keeping.camera:
        house_keeping.camera = None
    if obj == house_keeping.box:
        house_keeping.box = None
#    bpy.data.objects.remove(obj)
    # Remove all plot objects connected to the deleted geometry.
    for plot in stack_remove_list:
        __stack__.remove(plot)

bpy.types.Object.__del__ = delete_plot_object


# Add the needed deletable_object object attribute to the Blender light class.
setattr(bpy.types.Object, 'deletable_object', None)
