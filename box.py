# bounding_box.py
"""
Contains routines to generate the bounding box..

Created on Wed Mar 10 16:03:00 2021

@author: Simon Candelaresi
"""


'''
Test:
import numpy as np
import importlib
import blendaviz as blt
importlib.reload(blt.plot1d)
importlib.reload(blt)

n = 30
nt = 100
time = np.linspace(0, 100, nt)
# z = np.linspace(0, 6*np.pi, n)
z = np.linspace(0, 6*np.pi, n)[:, np.newaxis] + np.sin(time)/5
x = 3*np.cos(z)
y = 3*np.sin(z)
r = np.linspace(0.1, 0.5, 30)
r = r[:, np.newaxis]*np.linspace(1, 5, nt)
# r = np.linspace(0.3, 1, 100)
# r = r[np.newaxis, :]
# rotation_x = np.ones([x.shape[0], time.shape[0]])*time/10
pl = blt.plot(x, y, z, marker='cube', time=time, radius=r)

pl = blt.plot(x, y, z, marker='cube', radius=0.5, rotation_x=z, rotation_y=np.zeros_like(x), rotation_z=np.zeros_like(x))
colors = np.random.random([x.shape[0], 4])
pl = blt.plot(x, y, z, marker='cube')
pl = blt.plot(x, y, z, marker='cube', color=colors)
pl.z = np.linspace(0, 1, 5)
pl.plot()
'''

def bounding_box(lx, ly, lz, color='black', width=0.1):
    """
    Generate a bounding box for the plot.

    call signature:

    bounding_box(lx, ly, lz, color='black', width=1)

    Keyword arguments:

    *lx, ly, lz*:
      Size of the box in x, y and z.

    *color*:
      Color of the box as string 'red', character 'r' or RGBA value (1, 0, 0, 1).

    *width*:
      Width of the bounding box.

    Examples:
    """

    import inspect

    # Assign parameters to the BoundingBox object.
    bounding_box_return = BoundingBox()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(bounding_box_return, argument, argument_dict[argument])

    # Plot the bounding box.
    bounding_box_return.plot()
    return bounding_box_return



class BoundingBox(object):
    """
    BoundingBox class including the parameters and plotting function.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        self.lx = 1
        self.ly = 1
        self.lz = 1
        self.color = 'black'
        self.width = 0.1


    def plot(self):
        """
        Plot a as a bounding box.
        """

        import bpy
        import numpy as np
        from . import colors

        # Create the cube.
        bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(0.935306, 2.45305, 2.96555))

        # Delete existing meshes.
        if not self.marker_mesh is None:
            bpy.ops.object.select_all(action='DESELECT')
            if isinstance(self.marker_mesh, list):
                for marker_mesh in self.marker_mesh:
                    marker_mesh.select_set(True)
                    bpy.ops.object.delete()
            else:
                self.marker_mesh.select_set(True)
                bpy.ops.object.delete()
            self.marker_mesh = None

        # Delete existing materials.
        if not self.mesh_material is None:
            if isinstance(self.mesh_material, list):
                for mesh_material in self.mesh_material:
                    bpy.data.materials.remove(mesh_material)
            else:
                bpy.data.materials.remove(self.mesh_material)
            self.mesh_material = None

        # Create the bezier curve.
        if self.marker is None:
            # Transform color string into rgba.
            color_rgba = colors.make_rgba_array(self.color, 1)

            self.curve_data = bpy.data.curves.new('DataCurve', type='CURVE')
            self.curve_data.dimensions = '3D'
            self.curve_object = bpy.data.objects.new('ObjCurve', self.curve_data)

            # Set the origin to the last point.
            self.curve_object.location = tuple((self.x[-1, self.time_index],
                                                self.y[-1, self.time_index],
                                                self.z[-1, self.time_index]))

            # Add the rest of the curve.
            self.poly_line = self.curve_data.splines.new('POLY')
            self.poly_line.points.add(self.x.shape[0])
            for param in range(self.x.shape[0]):
                self.poly_line.points[param].co = (self.x[param, self.time_index] - self.x[-1, self.time_index],
                                                   self.y[param, self.time_index] - self.y[-1, self.time_index],
                                                   self.z[param, self.time_index] - self.z[-1, self.time_index],
                                                   0)

            # Add 3d structure.
            self.curve_data.splines.data.bevel_depth = self.radius[0, self.time_index]
            self.curve_data.splines.data.bevel_resolution = self.resolution
            self.curve_data.splines.data.fill_mode = 'FULL'

            # Set the material/color.
            self.mesh_material = bpy.data.materials.new('material')
            self.mesh_material.diffuse_color = color_rgba[0]
            self.mesh_material.roughness = self.roughness
            self.curve_object.active_material = self.mesh_material

            # Set the emission.
            if not self.emission is None:
                self.mesh_material.use_nodes = True
                node_tree = self.mesh_material.node_tree
                nodes = node_tree.nodes
                # Remove and Diffusive BSDF node.
                nodes.remove(nodes[1])
                node_emission = nodes.new(type='ShaderNodeEmission')
                # Change the input of the ouput node to emission.
                node_tree.links.new(node_emission.outputs['Emission'],
                                    nodes[0].inputs['Surface'])
                # Adapt emission and color.
                node_emission.inputs['Color'].default_value = color_rgba
                node_emission.inputs['Strength'].default_value = self.emission

            # Link the curve object with the scene.
            bpy.context.scene.collection.objects.link(self.curve_object)

        # Transform color string into rgb.
        color_rgba = colors.make_rgba_array(self.color, self.x.shape[0])

        # Plot the markers.
        if not self.marker is None:
            self.marker_mesh = []
        if self.marker == 'cone':
            for idx in range(self.x.shape[0]):
                bpy.ops.mesh.primitive_cone_add(location=(self.x[idx, self.time_index],
                                                          self.y[idx, self.time_index],
                                                          self.z[idx, self.time_index]),
                                                radius1=self.radius[idx, self.time_index],
                                                depth=2*self.radius[idx, self.time_index],
                                                rotation=(self.rotation_x[idx, self.time_index],
                                                          self.rotation_y[idx, self.time_index],
                                                          self.rotation_z[idx, self.time_index]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'cube':
            for idx in range(self.x.shape[0]):
                bpy.ops.mesh.primitive_cube_add(location=(self.x[idx, self.time_index],
                                                          self.y[idx, self.time_index],
                                                          self.z[idx, self.time_index]),
                                                size=self.radius[idx, self.time_index],
                                                rotation=(self.rotation_x[idx, self.time_index],
                                                          self.rotation_y[idx, self.time_index],
                                                          self.rotation_z[idx, self.time_index]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'cylinder':
            for idx in range(self.x.shape[0]):
                bpy.ops.mesh.primitive_cylinder_add(location=(self.x[idx, self.time_index],
                                                              self.y[idx, self.time_index],
                                                              self.z[idx, self.time_index]),
                                                    radius=self.radius[idx, self.time_index], depth=2*self.radius[idx, self.time_index],
                                                    rotation=(self.rotation_x[idx, self.time_index],
                                                              self.rotation_y[idx, self.time_index],
                                                              self.rotation_z[idx, self.time_index]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'ico_sphere':
            for idx in range(self.x.shape[0]):
                bpy.ops.mesh.primitive_ico_sphere_add(location=(self.x[idx, self.time_index],
                                                                self.y[idx, self.time_index],
                                                                self.z[idx, self.time_index]),
                                                      radius=self.radius[idx, self.time_index],
                                                      rotation=(self.rotation_x[idx, self.time_index],
                                                                self.rotation_y[idx, self.time_index],
                                                                self.rotation_z[idx, self.time_index]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'monkey':
            for idx in range(self.x):
                bpy.ops.mesh.primitive_monkey_add(location=(self.x[idx, self.time_index],
                                                            self.y[idx, self.time_index],
                                                            self.z[idx, self.time_index]),
                                                  size=self.radius[idx, self.time_index],
                                                  rotation=(self.rotation_x[idx, self.time_index],
                                                            self.rotation_y[idx, self.time_index],
                                                            self.rotation_z[idx, self.time_index]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'torus':
            for idx in range(self.x.shape[0]):
                bpy.ops.mesh.primitive_torus_add(location=(self.x[idx, self.time_index],
                                                           self.y[idx, self.time_index],
                                                           self.z[idx, self.time_index]),
                                                 major_radius=self.radius[idx, self.time_index],
                                                 minor_radius=0.25*self.radius[idx, self.time_index],
                                                 abso_major_rad=1.25*self.radius[idx, self.time_index],
                                                 abso_minor_rad=0.75*self.radius[idx, self.time_index],
                                                 rotation=(self.rotation_x[idx, self.time_index],
                                                           self.rotation_y[idx, self.time_index],
                                                           self.rotation_z[idx, self.time_index]))
                self.marker_mesh.append(bpy.context.object)
        if self.marker == 'uv_sphere':
            for idx in range(self.x.shape[0]):
                bpy.ops.mesh.primitive_uv_sphere_add(location=(self.x[idx, self.time_index],
                                                               self.y[idx, self.time_index],
                                                               self.z[idx, self.time_index]),
                                                     radius=self.radius[idx, self.time_index],
                                                     rotation=(self.rotation_x[idx, self.time_index],
                                                               self.rotation_y[idx, self.time_index],
                                                               self.rotation_z[idx, self.time_index]))
                self.marker_mesh.append(bpy.context.object)
        if isinstance(self.marker, bpy.types.Object):
            if self.marker.type == 'MESH':
                bpy.context.object.select = False
                self.marker.select = True
                for idx in range(self.x.shape[0]):
                    bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'})
                    bpy.context.object.location = (self.x[idx, self.time_index],
                                                   self.y[idx, self.time_index],
                                                   self.z[idx, self.time_index])
                    bpy.context.object.rotation_euler = (self.rotation_x[idx], self.rotation_y[idx], self.rotation_z[idx])
                    self.marker.select = False
                    self.marker_mesh.append(bpy.context.object)

        # Set the material and color.
        if not self.marker is None:
            color_is_array = False
            if isinstance(color_rgba, np.ndarray):
                if color_rgba.ndim == 2:
                    color_is_array = True

            if any([color_is_array,
                    isinstance(self.roughness, np.ndarray),
                    isinstance(self.emission, np.ndarray)]):
                self.mesh_material = []

                for idx in range(self.x.shape[0]):
                    self.mesh_material.append(bpy.data.materials.new('material'))

                    if color_is_array:
                        self.mesh_material[idx].diffuse_color = tuple(color_rgba[idx])
                    else:
                        self.mesh_material[idx].diffuse_color = color_rgba

                    if isinstance(self.roughness, np.ndarray):
                        self.mesh_material[idx].roughness = self.roughness[idx]
                    else:
                        self.mesh_material[idx].roughness = self.roughness

                    if isinstance(self.emission, np.ndarray):
                        self.mesh_material[idx].use_nodes = True
                        node_tree = self.mesh_material[idx].node_tree
                        nodes = node_tree.nodes
                        # Remove and Diffusive BSDF node.
                        nodes.remove(nodes[1])
                        node_emission = nodes.new(type='ShaderNodeEmission')
                        # Change the input of the ouput node to emission.
                        node_tree.links.new(node_emission.outputs['Emission'],
                                            nodes[0].inputs['Surface'])
                        # Adapt emission and color.
                        if color_is_array:
                            node_emission.inputs['Color'].default_value = tuple(color_rgba[idx])
                        else:
                            node_emission.inputs['Color'].default_value = color_rgba
                        node_emission.inputs['Strength'].default_value = self.emission[idx]

                    self.marker_mesh[idx].active_material = self.mesh_material[idx]
            else:
                self.mesh_material = bpy.data.materials.new('material')
                self.mesh_material.diffuse_color = color_rgba
                self.mesh_material.roughness = self.roughness

                if not self.emission is None:
                    self.mesh_material.use_nodes = True
                    node_tree = self.mesh_material.node_tree
                    nodes = node_tree.nodes
                    # Remove and Diffusive BSDF node.
                    nodes.remove(nodes[1])
                    node_emission = nodes.new(type='ShaderNodeEmission')
                    # Change the input of the ouput node to emission.
                    node_tree.links.new(node_emission.outputs['Emission'],
                                        nodes[0].inputs['Surface'])
                    # Adapt emission and color.
                    node_emission.inputs['Color'].default_value = color_rgba
                    node_emission.inputs['Strength'].default_value = self.emission

                for idx, mesh in enumerate(self.marker_mesh):
                    mesh.active_material = self.mesh_material

        # Group the meshes together.
        if not self.marker is None:
            for mesh in self.marker_mesh[::-1]:
                mesh.select_set(state=True)
            bpy.ops.object.join()
            self.marker_mesh = bpy.context.object
            self.marker_mesh.select_set(state=False)


        return 0
