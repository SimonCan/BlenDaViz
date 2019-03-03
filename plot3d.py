# plot3d.py
"""
Contains routines to three-dimensional plots.

Created on Wed Dec 16 10:34:00 2018

@author: Simon Candelaresi
"""

'''
Test:
import numpy as np
import importlib
import blender as blt
importlib.reload(blt.plot3d)
importlib.reload(blt)
x0 = np.linspace(-3, 3, 20)
y0 = np.linspace(-3, 3, 20)
z0 = np.linspace(-3, 3, 20)
x, y, z = np.meshgrid(x0, y0, z0, indexing='ij')
phi = np.exp(-(x**2+y**2+z**2))*np.cos(z)
alpha = np.sin(np.linspace(phi.min(), phi.max(), 100))
vol = vol.mesh(phi, x, y, z, alpha=)
vol.plot()
'''

# TODO:
# vol:
# - 1) Specify alpha values as 1xn or 2xn array.
# quiver:
# - 1) Pivot points.
# streamlines:
# - 1) Different metrics.

def vol(phi, x, y, z, emission=None, color_map=None):
    """
    Plot a 3d volume rendering of a scalar field.

    call signature:

    vol(phi, x, y, z, alpha=None, color_map=None)

    Keyword arguments:

    *phi*:
      Scalar field in 3d.

    *x, y, z*:
      1d arrays for the coordinates.

    *emission*:
      1d or 2d array of floats containing the emission values for phi.
      If specified as 2d array fo shape [2, n_emission] the emission[0, :] are the values
      phi which must be monotonously increasing and emission[1, ;] are their corresponding
      emission values.

    *color_map*:
      Color map for the values stored in the array 'c'.
      These are the same as in matplotlib.
    """

    import inspect

    # Assign parameters to the Volume objects.
    volume_return = Volume()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(volume_return, argument, argument_dict[argument])
    volume_return.plot()
    return volume_return


class Volume(object):
    """
    Volume class including the data, 3d texture and parameters.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        self.phi = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.emission = None
        self.color_map = None
        self.mesh_object = None
        self.mesh_material = None


    def plot(self):
        '''
        Plot the 3d texture.
        '''

        import bpy
        import numpy as np
        import matplotlib.cm as cm

        # Check the validity of the input arrays.
        if (not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray)
            or not isinstance(self.z, np.ndarray)):
            print("Error: x OR y OR z array invalid.")
            return -1
        if not isinstance(self.phi, np.ndarray):
            print("Error: phi must be numpy array.")
            return -1
        if not isinstance(self.emission, np.ndarray):
            print("Error: emission must be numpy array.")
            return -1

        # Using volumetric textures or voxels?

        # Delete existing meshes.
        if not self.mesh_object is None:
            bpy.ops.object.select_all(action='DESELECT')
            self.mesh_object.select = True
            bpy.ops.object.delete()
            self.mesh_object = None

        # Delete existing materials.
        if not self.mesh_material is None:
            bpy.data.materials.remove(self.mesh_material)

        # Create cuboid.
        bpy.ops.mesh.primitive_cube_add()
        self.mesh_object = bpy.context.object

        # Define the RGB value for each voxel.
        phi_max = np.max(self.phi)
        phi_min = np.min(self.phi)
        if self.color_map is None:
            self.color_map = cm.viridis
        pixels[0::3] = self.color_map((self.phi.flatten() - phi_min)/(phi_max - phi_min))[:, 0]
        pixels[1::3] = self.color_map((self.phi.flatten() - phi_min)/(phi_max - phi_min))[:, 1]
        pixels[2::3] = self.color_map((self.phi.flatten() - phi_min)/(phi_max - phi_min))[:, 2]

        # Define the emission for each voxel.
        emission = np.zeros_like(self.phi)
        if self.emission.ndim == 1:
            phi_emission = np.zeros([2, self.emission.size])
            phi_emission[0, :] = np.linspace(self.phi.min(), self.size.max(), self.emission.size)
            phi_emission[1, :] = self.emission
        else:
            phi_emission = self.emission

        for emission_idx in self.emission.shape[-1]-1:
            mask = self.phi >= phi_emission[0, emission_idx] and self.phi < phi_emission[0, emission_idx+1]
            weight_left = self.phi[mask] - phi_emission[0, emission_idx]
            weight_right = -self.phi[mask] + phi_emission[0, emission_idx+1]
            emission[mask] = (weight_left*phi_emission[1, emission_idx] + weight_right*phi_emission[1, emission_idx+1]) \
                             /(weight_left + weight_right)
        del(emission)

        # Assign a material to the cuboid.
        self.mesh_material = bpy.data.materials.new('MaterialMesh')
        self.mesh_material.use_nodes = True

        # Add the RGB and emission values to the material.
        nodes = node_tree.nodes
        # Remove diffusive BSDF node.
        nodes.remove(nodes[1])
        # Add the emission shader.
        node_emission = nodes.new(type='ShaderNodeEmission')
        # Link the emission output to the material output.
        node_tree.links.new(node_emission.outputs['Emission'],
                            nodes[0].inputs['Volume'])
        # Add the RGB source node.

        # Link the RGB output to the emission shader  color input.

        # Add the emission source node.

        # Link the emission output to the emission shader strength input.


        return 0
