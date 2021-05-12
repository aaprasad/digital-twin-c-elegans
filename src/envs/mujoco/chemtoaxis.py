""" swimmer: chemotaxis
- wrap env with chemotaxis task (with only 1 attractive/repulsive source)
"""

from lxml import etree
import numpy as np


def make_geom(worldbody, name, x, y, size, rgba):
    """ make a sphere
    Args:
        x, y: position
        size: radius
        rgba: color and transparency, e.g. '0 1 0 1' for green, '1 0 0 1' for red
    """
    body = etree.Element('body', attrib={'name': name, 'pos': '0 0 -0.1'})
    geom = etree.Element('geom', attrib={'density': '1000', 'pos': '{} {} {}'.format(x, y, size), 'size': '{}'.format(size), 'type': 'sphere', 'material': '', 'rgba': rgba})
    body.append(geom)
    worldbody.append(body)


def _make_model(xml_str):
    """ make a sphere representing 1 source """
    # create mjcf
    mjcf = etree.fromstring(xml_str, parser=etree.XMLParser(remove_blank_text=True))
    worldbody = mjcf.find('worldbody')
    # make sphere
    x = np.random.randint(-20, 20)
    y = np.sqrt(20 ** 2 - x ** 2)
    make_geom(worldbody, 'food', x=x, y=y, size=0.5, rgba='0 1 0 1')
    return mjcf


def chemotaxis(xml_str):
    mjcf = _make_model(xml_str=xml_str)
    return etree.tostring(mjcf, pretty_print=True)
