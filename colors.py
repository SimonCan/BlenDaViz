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
