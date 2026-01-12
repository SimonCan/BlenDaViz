# globals.py
"""
Contains some global variables for control.
"""

from typing import Optional, List, Any


class HouseKeeping:
    """
    Contains some global house keeping variables.
    """

    x_min: Optional[float]
    x_max: Optional[float]
    y_min: Optional[float]
    y_max: Optional[float]
    z_min: Optional[float]
    z_max: Optional[float]
    box: Optional[Any]  # blt.BoundingBox
    lights: List[Optional[Any]]  # List of bpy.types.Object
    camera: Optional[Any]  # bpy.types.Object

    def __init__(self) -> None:
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

        self.lights = [None for i in range(6)]

        self.camera = None
