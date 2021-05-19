""" swimmer: forage
- wrap env with forage task (with multiple attractive/repulsive sources)
"""

from .chemotaxis import make_source_geom
from lxml import etree
import numpy as np


def make_box(worldbody, name, x_pos, y_pos, x_size, y_size, z_size):
    """ make a body including a box geom and append it to worldbody
    Args:
        x_pos, y_pos: position of the box geom
            - z_pos is the same as z_size so that the geom is placed on the plane
        x_size, y_size, z_size: half-sizes of the box along the X, Y and Z axes
    """
    body = etree.Element('body', attrib={'name': name, 'pos': '0 0 -0.1'})
    geom = etree.Element(
        'geom',
        attrib={
            'name': '{}_geom'.format(name), 'density': '1000', 'type': 'box', 'material': '', 'rgba': '0.5 0.5 0.5 1',
            'pos': '{} {} {}'.format(x_pos, y_pos, z_size),
            'size': '{} {} {}'.format(x_size, y_size, z_size)
        }
    )
    body.append(geom)
    worldbody.append(body)


def make_perimeter(worldbody, width, box_width, box_height):
    """ make a square perimeter wall
    - the wall has mass, otherwise there is a bug because it can't compute center of mass
    Args:
        width: half-size of the square's width
        box_width: half-size of the wall's width
        box_height: half-size of the wall's height
    """
    # 2 parallel sides
    make_box(worldbody, 'perimeter1', x_pos=box_width, y_pos=width, x_size=width, y_size=box_width, z_size=box_height)
    make_box(worldbody, 'perimeter2', x_pos=-box_width, y_pos=-width, x_size=width, y_size=box_width, z_size=box_height)
    # the other 2 paralell sides
    make_box(worldbody, 'perimeter3', x_pos=width, y_pos=-box_width, x_size=box_width, y_size=width, z_size=box_height)
    make_box(worldbody, 'perimeter4', x_pos=-width, y_pos=box_width, x_size=box_width, y_size=width, z_size=box_height)


def _make_model(xml_str, perimeter_width):
    """ make sphere and perimeter wall
    Args:
        perimeter_width: half-size of square perimeter wall's width
    """
    # create mjcf
    mjcf = etree.fromstring(xml_str, parser=etree.XMLParser(remove_blank_text=True))
    worldbody = mjcf.find('worldbody')
    # make sphere: green for food, red for trap
    for i in range(5):
        for rgba, name in zip(['0 1 0 1', '1 0 0 1'], ['food{}'.format(i + 1), 'trap{}'.format(i + 1)]):
            x = np.random.randint(-perimeter_width, perimeter_width)
            y = np.random.randint(-perimeter_width, perimeter_width)
            make_source_geom(worldbody, name, x=x, y=y, rgba=rgba, radius=0.5)
    # make perimeter wall
    make_perimeter(worldbody, width=perimeter_width, box_width=0.5, box_height=0.5)
    return mjcf


def forage(xml_str, perimeter_width):
    mjcf = _make_model(xml_str=xml_str, perimeter_width=perimeter_width)
    return etree.tostring(mjcf, pretty_print=True)
