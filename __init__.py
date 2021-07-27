#
# The __init__ file is used not only to import the sub-modules, but also to
# set everything up properly.
#

# Install missing python packages using:
# python3.7m -m pip install [LIB_NAME]

import bpy

# Load sub-modules.
from .generic import *
from .plot1d import *
from .plot2d import *
from .plot3d import *
from .streamlines3d import *
from .matplotlib_bridge import *
from .colors import *
from .vectors import *
from .globals import *
from .box import *
from .lights import *
from .camera import *
#from .seeds import *


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
        return context.active_object is not None

    def execute(self, context):
        stack_remove_list = []
        # Find the plot objects that will be removed from the stack.
        for obj in context.selected_objects:
            for plot in __stack__:
                stack_obj = plot.deletable_object
                if stack_obj == obj:
                    stack_remove_list.append(plot)
            bpy.data.objects.remove(obj)
        # Remove all plot objects connected to the deleted geometry.
        for plot in stack_remove_list:
            __stack__.remove(plot)
        return {'FINISHED'}

def register_delete_override():
    bpy.utils.register_class(DeleteOverride)

def unregister_delete_override():
    bpy.utils.unregister_class(DeleteOverride)

register_delete_override()
