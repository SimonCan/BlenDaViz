"""
Contains routines to generate the bounding box.
"""

from typing import Optional, List, Any, Tuple
import numpy.typing as npt


def bounding_box(extrema: Optional[Tuple[float, float, float, float, float, float]] = None) -> 'BoundingBox':
    """
    Plot a bounding box with a given size.

    Signature:

    bounding_box(extrema)

    Parameters
    ----------
    extrema:  If none given, use globals.
        Array of length 6 with components [x_min, x_max, y_min, y_max, z_min, z_max].

    Returns
    -------
    Class containing the bounding box.

    Examples
    --------
    >>> import numpy as np
    >>> import blendaviz as blt
    >>> extrema = np.array([0, 1, 0, 2, 0, 3])
    >>> blt.bounding_box(extrema)
    """

    import inspect

    # Assign parameters to the PathLine objects.
    bounding_box_return = BoundingBox()
    argument_dict = inspect.getargvalues(inspect.currentframe()).locals
    for argument in argument_dict:
        setattr(bounding_box_return, argument, argument_dict[argument])

    # Plot the bounding box.
    bounding_box_return.plot()
    return bounding_box_return



class BoundingBox:
    """
    Bounding box class including the splinces, parameters and plotting function.
    """

    extrema: Optional[Tuple[float, float, float, float, float, float]]
    curve_data: Optional[List[Any]]  # List of bpy.types.Curve
    curve_object: Optional[Any]  # bpy.types.Object or List of them
    poly_line: Optional[List[Any]]  # List of bpy.types.Spline
    deletable_object: Optional[Any]  # bpy.types.Object

    def __init__(self) -> None:
        """
        Fill members with default values.
        """

        import blendaviz as blt

        # Define the members that can be seen by the user.
        self.extrema = None
        self.curve_data = None
        self.curve_object = None
        self.poly_line = None
        self.deletable_object = None

        # Add the plot to the stack.
        blt.plot_stack.append(self)


    def plot(self) -> int:
        """
        Plot the bounding box.
        """

        import bpy
        # from blendaviz import colors

        # Check if extrema are given.
        if self.extrema is None:
            self.get_extrema()

        # Delete existing curve.
        if not self.curve_data is None:
            for curve_data in self.curve_data:
                bpy.data.curves.remove(curve_data)

        # Initialize the list of curve data and object.
        self.curve_data = []
        self.curve_object = []
        self.poly_line = []

        # Create the bezier curve.
        self.curve_data.append(bpy.data.curves.new('DataCurve', type='CURVE'))
        self.curve_data[-1].dimensions = '3D'
        self.curve_object.append(bpy.data.objects.new('ObjCurve', self.curve_data[-1]))
        # Set the origin.
        self.curve_object[-1].location = tuple((self.extrema[0], self.extrema[2], self.extrema[4]))
        # Add the rest of the curve.
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[1] - self.extrema[0],
                                           self.extrema[2] - self.extrema[2],
                                           self.extrema[4] - self.extrema[4], 0)
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[0] - self.extrema[0],
                                           self.extrema[3] - self.extrema[2],
                                           self.extrema[4] - self.extrema[4], 0)
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[0] - self.extrema[0],
                                           self.extrema[2] - self.extrema[2],
                                           self.extrema[5] - self.extrema[4], 0)

        # Create the bezier curve.
        self.curve_data.append(bpy.data.curves.new('DataCurve', type='CURVE'))
        self.curve_data[-1].dimensions = '3D'
        self.curve_object.append(bpy.data.objects.new('ObjCurve', self.curve_data[-1]))
        # Set the origin.
        self.curve_object[-1].location = tuple((self.extrema[1], self.extrema[3], self.extrema[4]))
        # Add the rest of the curve.
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[2] - self.extrema[1],
                                           self.extrema[3] - self.extrema[3],
                                           self.extrema[4] - self.extrema[4], 0)
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[1] - self.extrema[1],
                                           self.extrema[2] - self.extrema[3],
                                           self.extrema[4] - self.extrema[4], 0)
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[1] - self.extrema[1],
                                           self.extrema[3] - self.extrema[3],
                                           self.extrema[5] - self.extrema[4], 0)

        # Create the bezier curve.
        self.curve_data.append(bpy.data.curves.new('DataCurve', type='CURVE'))
        self.curve_data[-1].dimensions = '3D'
        self.curve_object.append(bpy.data.objects.new('ObjCurve', self.curve_data[-1]))
        # Set the origin.
        self.curve_object[-1].location = tuple((self.extrema[0], self.extrema[3], self.extrema[5]))
        # Add the rest of the curve.
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[1] - self.extrema[0],
                                           self.extrema[3] - self.extrema[3],
                                           self.extrema[5] - self.extrema[5], 0)
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[0] - self.extrema[0],
                                           self.extrema[2] - self.extrema[3],
                                           self.extrema[5] - self.extrema[5], 0)
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[0] - self.extrema[0],
                                           self.extrema[3] - self.extrema[3],
                                           self.extrema[4] - self.extrema[5], 0)

        # Create the bezier curve.
        self.curve_data.append(bpy.data.curves.new('DataCurve', type='CURVE'))
        self.curve_data[-1].dimensions = '3D'
        self.curve_object.append(bpy.data.objects.new('ObjCurve', self.curve_data[-1]))
        # Set the origin.
        self.curve_object[-1].location = tuple((self.extrema[1], self.extrema[2], self.extrema[5]))
        # Add the rest of the curve.
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[0] - self.extrema[1],
                                           self.extrema[2] - self.extrema[2],
                                           self.extrema[5] - self.extrema[5], 0)
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[1] - self.extrema[1],
                                           self.extrema[3] - self.extrema[2],
                                           self.extrema[5] - self.extrema[5], 0)
        self.poly_line.append(self.curve_data[-1].splines.new('POLY'))
        self.poly_line[-1].points.add(1)
        self.poly_line[-1].points[0].co = (self.extrema[1] - self.extrema[1],
                                           self.extrema[2] - self.extrema[2],
                                           self.extrema[4] - self.extrema[5], 0)

        # Link the curve object with the scene.
        for curve_object in self.curve_object:
            bpy.context.scene.collection.objects.link(curve_object)

        # Group the splines together.
        for curve_object in self.curve_object[::-1]:
            curve_object.select_set(state=True)
            bpy.context.view_layer.objects.active = curve_object
        bpy.ops.object.join()
        self.curve_object = self.curve_object[0]
        # Make this box the object to be deleted.
        self.deletable_object = self.curve_object

        return 0


    def get_extrema(self) -> None:
        """
        Get the extrema from the global structure.
        """

        import blendaviz as blt

        self.extrema = (blt.house_keeping.x_min, blt.house_keeping.x_max,
                        blt.house_keeping.y_min, blt.house_keeping.y_max,
                        blt.house_keeping.z_min, blt.house_keeping.z_max)
