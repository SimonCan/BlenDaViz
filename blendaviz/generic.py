# generic_plot.py
"""
The generic plotting routine from which all others inherit.
"""

class GenericPlot():
    """
    Generic plotting class with minimal functionality.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        import bpy

        # Define the members that can be seen by the user.
        self.time = None

        ## Set the handler function for frame changes (time).
        #bpy.app.handlers.frame_change_pre.append(self.time_handler)


    def plot(self):
        """
        Plot nothing.
        """


    def time_handler(self, scene, depsgraph):
        """
        Function to be called whenever any Blender animation functions are used.
        Updates the plot according to the function specified.
        """

        if not self.time is None:
            self.plot()


    def update_globals(self):
        """
        Update the global variables.
        """


    def update_extrema_x(self):
        """
        Update the domain in the x-direction.
        """

        import blendaviz as blt

        if blt.house_keeping.x_min is None:
            blt.house_keeping.x_min = self.x.min()
        elif self.x.min() < blt.house_keeping.x_min:
            blt.house_keeping.x_min = self.x.min()
        if blt.house_keeping.x_max is None:
            blt.house_keeping.x_max = self.x.max()
        elif self.x.max() > blt.house_keeping.x_max:
            blt.house_keeping.x_max = self.x.max()


    def update_extrema_y(self):
        """
        Update the domain in the x-direction.
        """

        import blendaviz as blt

        if blt.house_keeping.y_min is None:
            blt.house_keeping.y_min = self.y.min()
        elif self.y.min() < blt.house_keeping.y_min:
            blt.house_keeping.y_min = self.y.min()
        if blt.house_keeping.y_max is None:
            blt.house_keeping.y_max = self.y.max()
        elif self.y.max() > blt.house_keeping.y_max:
            blt.house_keeping.y_max = self.y.max()


    def update_extrema_z(self):
        """
        Update the domain in the x-direction.
        """

        import blendaviz as blt

        if blt.house_keeping.z_min is None:
            blt.house_keeping.z_min = self.z.min()
        elif self.z.min() < blt.house_keeping.z_min:
            blt.house_keeping.z_min = self.z.min()
        if blt.house_keeping.z_max is None:
            blt.house_keeping.z_max = self.z.max()
        elif self.z.max() > blt.house_keeping.z_max:
            blt.house_keeping.z_max = self.z.max()


    def object_reference_valid(self, obj):
        """
        Verify that the reference to the object obj is valid.
        This is useful after the user manually deletes objects
        and we try to refer to it.
        """

        try:
            dir(obj)
            valid = True
        except:
            valid = False

        return valid
