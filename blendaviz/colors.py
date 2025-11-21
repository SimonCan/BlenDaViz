# colors.py
"""
Contains routines to handle colors.
"""

def string_to_rgba(color_string):
    """
    Converts a color string or character into an rgb value.

    Signature:

    string_to_rgb(color_string):

    Parameters
    ----------
    color_string:  Any valid color string or character.
    """

    if color_string == 'b':
        rgba = (0, 0, 1, 1)
    if color_string == 'g':
        rgba = (0, 1, 0, 1)
    if color_string == 'r':
        rgba = (1, 0, 0, 1)
    if color_string == 'c':
        rgba = (0, 1, 1, 1)
    if color_string == 'm':
        rgba = (1, 0, 1, 1)
    if color_string == 'y':
        rgba = (1, 1, 0, 1)
    if color_string == 'k':
        rgba = (0, 0, 0, 1)
    if color_string == 'w':
        rgba = (1, 1, 1, 1)

    if color_string == 'blue':
        rgba = (0, 0, 1, 1)
    if color_string == 'green':
        rgba = (0, 1, 0, 1)
    if color_string == 'red':
        rgba = (1, 0, 0, 1)
    if color_string == 'cyan':
        rgba = (0, 1, 1, 1)
    if color_string == 'magenta':
        rgba = (1, 0, 1, 1)
    if color_string == 'yellow':
        rgba = (1, 1, 0, 1)
    if color_string == 'black':
        rgba = (0, 0, 0, 1)
    if color_string == 'white':
        rgba = (1, 1, 1, 1)

    return rgba


def make_rgba_array(color, length, color_map=None, vmin=None, vmax=None):
    """
    Creates an rgb array for the given color, which can be rgb, scalar array
    or string (array).

    Signature:

    make_rgba_array(color, length, color_map, vmin, vmax):

    Parameters
    ----------
    color:  Any valid color string, character, tuple, array or list.

    length:  Length of the data array (int).

    color_map:  Color map for the values.
        These are the same as in matplotlib.

    vmin, vmax:  Minimum and maximum values for the colormap. If not specify,
        determine from the input arrays.
    """

    import numpy as np
    from matplotlib import cm

    # Assign the colormap to the rgb values.
    if isinstance(color, np.ndarray):
        if color.ndim == 1:
            if color_map is None:
                color_map = cm.viridis
            if vmin is None:
                vmin = color.min()
            if vmax is None:
                vmax = color.max()
            color_rgba = color_map((color - vmin)/(vmax - vmin))[:, :]
        if color.ndim == 2:
            if color.shape[1] == 3:
                color_rgba = np.ones([color.shape[0], 4])
                color_rgba[:, :3] = color
            else:
                color_rgba = color

    # Transform list of color string into rgba.
    if isinstance(color, list):
        if len(color) != length:
            return -1
        color_rgba = np.ones([len(color), 4])
        for color_index, color_string in enumerate(color):
            if isinstance(color_string, str):
                color_rgba[color_index, :] = string_to_rgba(color_string)
            elif isinstance(color_string, (tuple, list)):
                color_rgba[color_index, :len(color_string)] = color_string
        print(color_rgba.shape, color_rgba)

    # Transform single color string into color array.
    if isinstance(color, str):
        color_rgba = np.array(string_to_rgba(color))

    # Transform single color tuple into color array.
    if isinstance(color, tuple):
        if len(color) == 3:
            color_rgba = color + (1,)
        else:
            color_rgba = color

    return color_rgba
