""" swimmer: forage
- wrap env with forage task
"""

from lxml import etree
import random


def _make_geom(mjcf, name, x, y, size, rgba):
    """ make a sphere
    Args:
        x, y: position
        size: radius
        rgba: color and transparency, e.g. '0 1 0 1' for green, '1 0 0 1' for red
    """
    body = etree.Element('body', attrib={'name': name, 'pos': '{} {} {}'.format(x, y, size - 0.1)})
    geom = etree.Element('geom', attrib={'density': '0', 'size': '{}'.format(size), 'type': 'sphere', 'material': '', 'rgba': rgba})
    body.append(geom)
    worldbody = mjcf.find('worldbody')
    worldbody.append(body)


def make_perimeter(mjcf, name, width, box_width, box_height):
    """ make a square perimeter wall
    - the wall has mass, otherwise there is a bug because it can't compute center of mass
    Args:
        width: half-size of the square's width
        box_width: half-size of the wall's width
        box_height: half-size of the wall's height
    """
    body = etree.Element('body', attrib={'name': name, 'pos': '0 0 -0.1'})
    # 2 parallel sides
    geom = etree.Element(
        'geom',
        attrib={
            'density': '1000', 'type': 'box', 'material': '', 'rgba': '0.5 0.5 0.5 1',
            'pos': '{} {} {}'.format(box_width, width, box_height),
            'size': '{} {} {}'.format(width, box_width, box_height)
        }
    )
    body.append(geom)
    geom = etree.Element(
        'geom',
        attrib={
            'density': '1000', 'type': 'box', 'material': '', 'rgba': '0.5 0.5 0.5 1',
            'pos': '{} {} {}'.format(-box_width, -width, box_height),
            'size': '{} {} {}'.format(width, box_width, box_height)
        }
    )
    body.append(geom)
    # the other 2 paralell sides
    geom = etree.Element(
        'geom',
        attrib={
            'density': '1000', 'type': 'box', 'material': '', 'rgba': '0.5 0.5 0.5 1',
            'pos': '{} {} {}'.format(width, -box_width, box_height),
            'size': '{} {} {}'.format(box_width, width, box_height)
        }
    )
    body.append(geom)
    geom = etree.Element(
        'geom',
        attrib={
            'density': '1000', 'type': 'box', 'material': '', 'rgba': '0.5 0.5 0.5 1',
            'pos': '{} {} {}'.format(-width, box_width, box_height),
            'size': '{} {} {}'.format(box_width, width, box_height)
        }
    )
    body.append(geom)
    # add to worldbody
    worldbody = mjcf.find('worldbody')
    worldbody.append(body)


def _make_model(xml_str, perimeter_width):
    """ make sphere and perimeter wall
    Args:
        perimeter_width: half-size of square perimeter wall's width
    """
    # create mjcf
    mjcf = etree.fromstring(xml_str, parser=etree.XMLParser(remove_blank_text=True))
    # add green sphere as food, red sphere as trap
    for i in range(5):
        for rgba, name in zip(['0 1 0 1', '1 0 0 1'], ['food{}'.format(i + 1), 'trap{}'.format(i + 1)]):
            x = random.randint(-perimeter_width, perimeter_width)
            y = random.randint(-perimeter_width, perimeter_width)
            _make_geom(mjcf=mjcf, name=name, x=x, y=y, size=0.5, rgba=rgba)
    # make perimeter wall
    make_perimeter(mjcf=mjcf, name='perimeter', width=perimeter_width, box_width=0.5, box_height=0.5)
    return mjcf


def forage(xml_str, perimeter_width):
    mjcf = _make_model(xml_str=xml_str, perimeter_width=perimeter_width)
    return etree.tostring(mjcf, pretty_print=True)
