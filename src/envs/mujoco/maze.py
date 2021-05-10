""" swimmer: maze
- wrap env with maze task
"""

from .forage import make_perimeter
from lxml import etree


def _make_box(mjcf, name, x, y, x_size, y_size, z_size):
    """ make a box
    Args:
        x_size, y_size, z_size: half-sizes of the box along the X, Y and Z axes
    """
    body = etree.Element('body', attrib={'name': name, 'pos': '{} {} {}'.format(x, y, z_size - 0.1)})
    geom = etree.Element('geom', attrib={'density': '1000', 'size': '{} {} {}'.format(x_size, y_size, z_size), 'type': 'box', 'material': '', 'rgba': '0.5 0.5 0.5 1'})
    body.append(geom)
    worldbody = mjcf.find('worldbody')
    worldbody.append(body)


def _make_model(xml_str, perimeter_width, box_width):
    """ make a maze with perimeter wall and other walls
    Args:
        perimeter_width: half-size of square perimeter wall's width
        box_width: half-size of wall's width (thickness)
    """
    # create mjcf
    mjcf = etree.fromstring(xml_str, parser=etree.XMLParser(remove_blank_text=True))
    # make perimeter wall
    make_perimeter(mjcf=mjcf, name='perimeter', width=perimeter_width, box_width=box_width, box_height=0.5)
    # make wall
    _make_box(mjcf=mjcf, name='wall1', x=-1.5, y=2, x_size=4, y_size=box_width, z_size=0.5)
    _make_box(mjcf=mjcf, name='wall2', x=1.5, y=-2, x_size=4, y_size=box_width, z_size=0.5)
    return mjcf


def maze(xml_str, perimeter_width, box_width):
    mjcf = _make_model(xml_str=xml_str, perimeter_width=perimeter_width, box_width=box_width)
    return etree.tostring(mjcf, pretty_print=True)
