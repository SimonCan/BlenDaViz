# generic_plot.py
"""
The generic plotting routine from which all others inherit.
"""

class GenericPlot(object):
    """
    Generic plotting class with minimal functionality.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        import bpy

        # Define the members that can be seen by the user.
        self. time = None

        # Define the locally used time-independent data and parameters.

        # Set the handler function for frame changes (time).
        bpy.app.handlers.frame_change_pre.append(self.time_handler)


    def __del__(self):
        """
        Delete geometry and materials and remove time handler
        and plot object from the stack.
        """

        plot("generic")
        pass


    def plot(self):
        """
        Plot nothing.
        """

        pass


    def time_handler(self, scene, depsgraph):
        """
        Function to be called whenever any Blender animation functions are used.
        Updates the plot according to the function specified.
        """

        if not self.time is None:
            self.plot()
        else:
            pass
