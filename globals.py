# globals.py
"""
Contains some global variables for control.
"""


class HouseKeeping(object):
    """
    Contains some global house keeping variables.
    """

    def __init__(self):
        """
        Fill members with default values.
        """

        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None

        self.box = None

        self.lights = None

        self.camera = None
