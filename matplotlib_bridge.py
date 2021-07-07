# matplotlib_bridge.py
"""
Contains classes for the embedding of matplotlib plots in BlenDaViz.
"""


'''
Test:
import sys
sys.path.append('~/codes/blendaviz')
import numpy as np
import importlib
import matplotlib.pyplot as plt
import blendaviz as blt
importlib.reload(blt.matplotlib_bridge)
importlib.reload(blt)

x = np.linspace(0, 5, 1000)
fig = plt.figure()
plt.plot(x, np.sin(x), color='g')
plt.title("test")

mbl = blt.mpl_figure_to_blender(fig)
'''

def mpl_figure_to_blender(figure, dpi=300, position=None, normal=None):
    """
    Plot a Matplotlib figure into blender.

    call signature:

    mesh(figure, dpi=300, corners=None)

    Keyword arguments:

    *figure*:
      Matplotlib figure from your plot.

    *dpi*:
      Resolution in dots per inch.

    *position*:
      Lower left corner for positioning.

    *normal*:
      Normal vector of the plane.

    Examples:
    """

    import inspect

    # Assign parameters to the Mesh objects.
    mpl_embedding_return = MPLEmbedding()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(mpl_embedding_return, argument, argument_dict[argument])
    mpl_embedding_return.plot()
    return mpl_embedding_return


class MPLEmbedding(object):
    """
    Surface class including the vertices, surfaces, parameters and plotting function.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        import blendaviz as blt

        # Define the members that can be seen by the user.
        self.figure = None
        self.dpi = 300
        self.position = None

        self.mesh_data = None
        self.mesh_object = None
        self.mesh_material = None
        self.deletable_object = None

        # Add the plot to the stack.
        blt.__stack__.append(self)


    def plot(self):
        """
        Plot the Matplotlib figure.
        """

        import bpy
        import io
        from PIL import Image
        import matplotlib.pyplot as plt
        import numpy as np

        # Check the 3d figure position and normal.
        if self.position is None:
            self.position = np.array([0, 0, 0])
        if self.normal is None:
            self.normal = np.array([0, 0, 1])
        self.position = np.array(self.position)
        self.normal = np.array(self.normal)
        
        # Delete existing meshes.
        if not self.mesh_object is None:
            bpy.ops.object.select_all(action='DESELECT')
            self.mesh_object.select_set(state=True)
            bpy.ops.object.delete()
            self.mesh_object = None

        # Delete existing materials.
        if not self.mesh_material is None:
            bpy.data.materials.remove(self.mesh_material)

        # Create plane.
        bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False,
                                         location=self.position)
        self.mesh_object = bpy.context.object

        # Resize the plane to match the plot size.
        bpy.ops.transform.resize(value=(self.figure.get_size_inches()[0],
                                        self.figure.get_size_inches()[1], 1),
                                mirror=True)

        # Orient the plane following the normal vector.
        rotation = np.zeros(3)
        rotation[0] = np.arcsin(self.normal[0]/np.sqrt(np.sum(self.normal**2)))
        rotation[1] = np.arcsin(self.normal[1]/np.sqrt(np.sum(self.normal**2)))
        bpy.ops.transform.rotate(value=rotation[0], orient_axis='X')
        bpy.ops.transform.rotate(value=rotation[1], orient_axis='Y')

        # Create the png image from the figure.
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, transparent=False)
        buffer.seek(0)
        im = Image.open(buffer)
        pixels = np.reshape(list(im.getdata()), [im.height, im.width, 4])[::-1, :, :]
        
        # Assign a material to the surface.
        self.mesh_data = bpy.context.object.data
        self.mesh_material = bpy.data.materials.new('MaterialMesh')
        self.mesh_data.materials.append(self.mesh_material)
        
        # Assign image texture to mesh.
        mesh_image = bpy.data.images.new('ImageMesh', im.width, im.height)
        print(np.array(pixels).shape)
        mesh_image.pixels = np.array(pixels).flatten()
        
        # Assign the texture to the material.
        self.mesh_material.use_nodes = True
        self.mesh_texture = self.mesh_material.node_tree.nodes.new('ShaderNodeTexImage')
        self.mesh_texture.image = mesh_image
        links = self.mesh_material.node_tree.links
        links.new(self.mesh_texture.outputs[0],
                  self.mesh_material.node_tree.nodes.get("Principled BSDF").inputs[0])
        
        buffer.close()

        # Make the mesh the deletable object.
        self.deletable_object = self.mesh_object

#        self.update_globals()

        return 0

#
#    def update_globals(self):
#        """
#        Update the extrema.
#        """
#
#        import blendaviz as blt
#        import numpy as np

#        if blt.house_keeping.x_min is None:
#            blt.house_keeping.x_min = np.min(self.corners[:, 0])
#        elif np.min(self.corners[:, 0]) < blt.house_keeping.x_min:
#            blt.house_keeping.x_min = np.min(self.corners[:, 0])
#        if blt.house_keeping.x_max is None:
#            blt.house_keeping.x_max = np.max(self.corners[:, 0])
#        elif np.max(self.corners[:, 0]) > blt.house_keeping.x_max:
#            blt.house_keeping.x_max = np.max(self.corners[:, 0])
#
#        if blt.house_keeping.y_min is None:
#            blt.house_keeping.y_min = np.min(self.corners[:, 1])
#        elif np.min(self.corners[:, 1]) < blt.house_keeping.y_min:
#            blt.house_keeping.y_min = np.min(self.corners[:, 1])
#        if blt.house_keeping.y_max is None:
#            blt.house_keeping.y_max = np.max(self.corners[:, 1])
#        elif np.max(self.corners[:, 1]) > blt.house_keeping.y_max:
#            blt.house_keeping.y_max = np.max(self.corners[:, 1])
#
#        if blt.house_keeping.z_min is None:
#            blt.house_keeping.z_min = np.min(self.corners[:, 2])
#        elif np.min(self.corners[:, 2]) < blt.house_keeping.z_min:
#            blt.house_keeping.z_min = np.min(self.corners[:, 2])
#        if blt.house_keeping.z_max is None:
#            blt.house_keeping.z_max = np.max(self.corners[:, 2])
#        elif np.max(self.corners[:, 2]) > blt.house_keeping.z_max:
#            blt.house_keeping.z_max = np.max(self.corners[:, 2])
#
#        if blt.house_keeping.box is None:
#            blt.house_keeping.box = blt.bounding_box()
#        else:
#            blt.house_keeping.box.get_extrema()
#            blt.house_keeping.box.plot()

#        # Add some light.
#        blt.adjust_lights()




