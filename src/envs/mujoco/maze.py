""" swimmer: maze
- wrap env with maze task
"""

from .forage import make_perimeter, make_box
from lxml import etree


def _make_model(xml_str, perimeter_width, box_width):
    """ make a maze with perimeter wall and other walls
    Args:
        perimeter_width: half-size of square perimeter wall's width
        box_width: half-size of wall's width (thickness)
    """
    # create mjcf
    mjcf = etree.fromstring(xml_str, parser=etree.XMLParser(remove_blank_text=True))
    worldbody = mjcf.find('worldbody')
    # make perimeter wall
    make_perimeter(worldbody, width=perimeter_width, box_width=box_width, box_height=0.5)
    # make wall
    make_box(worldbody, 'wall1', x_pos=-1.5, y_pos=2, x_size=4, y_size=box_width, z_size=0.5)
    make_box(worldbody, 'wall2', x_pos=1.5, y_pos=-2, x_size=4, y_size=box_width, z_size=0.5)
    return mjcf


def maze(xml_str, perimeter_width, box_width):
    mjcf = _make_model(xml_str=xml_str, perimeter_width=perimeter_width, box_width=box_width)
    return etree.tostring(mjcf, pretty_print=True)
