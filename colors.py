# colors.py
"""
Contains routines to handle colors.

Created on Mon Aug 27 18:00:35 2018

@author: Simon Candelaresi
"""

def string_to_rgb(color_string):
    """
    Converts a color string or character into an rgb value.

    call signature:

    string_to_rgb(color_string):

    Keyword arguments:

    *color_string*:
      Any valid color string or character.
    """

    if color_string == 'b':
        rgb = (0, 0, 1)
    if color_string == 'g':
        rgb = (0, 1, 0)
    if color_string == 'r':
        rgb = (1, 0, 0)
    if color_string == 'c':
        rgb = (0, 1, 1)
    if color_string == 'm':
        rgb = (1, 0, 1)
    if color_string == 'y':
        rgb = (1, 1, 0)
    if color_string == 'k':
        rgb = (0, 0, 0)
    if color_string == 'w':
        rgb = (1, 1, 1)

    if color_string == 'blue':
        rgb = (0, 0, 1)
    if color_string == 'green':
        rgb = (0, 1, 0)
    if color_string == 'red':
        rgb = (1, 0, 0)
    if color_string == 'cyan':
        rgb = (0, 1, 1)
    if color_string == 'magenta':
        rgb = (1, 0, 1)
    if color_string == 'yellow':
        rgb = (1, 1, 0)
    if color_string == 'black':
        rgb = (0, 0, 0)
    if color_string == 'white':
        rgb = (1, 1, 1)

    return rgb


def make_rgb_array(color, length, color_map=None, vmin=None, vmax=None):
    """
    Creates an rgb array for the civen color, which can be rgb, scalar array
    or string (array).

    call signature:

    make_rgb_array(color, length, color_map, vmin, vmax):

    Keyword arguments:

    *color*:
      Any valid color string or character.

    *length*:
      Length of the data array (int).

    *color_map*:
      Color map for the values.
      These are the same as in matplotlib.

    *vmin, vmax*
      Minimum and maximum values for the colormap. If not specify, determine
      from the input arrays.
    """

    import numpy as np
    import matplotlib.cm as cm

    # Assign the colormap to the rgb values.
    if isinstance(color, np.ndarray):
        if color.ndim == 1:
            if color_map is None:
                color_map = cm.viridis
            if vmin is None:
                vmin = color.min()
            if vmax is None:
                vmax = color.max()
            color_rgb = np.zeros([length, 3])
            color_rgb = color_map((color - vmin)/(vmax - vmin))[:, :3]

    # Copy rgb values if given.
    if isinstance(color, np.ndarray):
        if color.ndim == 2:
            color_rgb = color

    # Transform color string into rgb.
    if isinstance(color, list):
        if len(color) != length:
            return -1
        color_rgb = np.zeros([len(color), 3])
        for color_index, color_string in enumerate(color):
            if isinstance(color_string, str):
                color_rgb[color_index, :] = string_to_rgb(color_string)
            elif len(color_string) == 3:
                color_rgb[color_index, :] = color[color_index]
    else:
        if isinstance(color, str):
            color_rgb = np.zeros([1, 3])
            color_rgb[0] = np.array(string_to_rgb(color))

    return color_rgb
