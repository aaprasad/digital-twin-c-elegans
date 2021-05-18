""" swimmer: chemotaxis
- wrap env with chemotaxis task (with only 1 attractive/repulsive source)
"""

from lxml import etree


def make_source_geom(worldbody, name, x, y, rgba, radius, half_length=0.1):
    """ make a cylinder
    Args:
        x, y: position
        rgba: color and transparency, e.g. '0 1 0 1' for green, '1 0 0 1' for red
        radius, half_length: cylinder params
    """
    body = etree.Element('body', attrib={'name': name, 'pos': '0 0 -0.1'})
    geom = etree.Element(
        'geom',
        attrib={
            'density': '1000', 'type': 'cylinder', 'material': '', 'rgba': rgba,
            'pos': '{} {} {}'.format(x, y, half_length), 'size': '{} {}'.format(radius, half_length)
        }
    )
    body.append(geom)
    worldbody.append(body)


def _make_model(xml_str, x, y):
    """ make a sphere representing 1 source """
    # create mjcf
    mjcf = etree.fromstring(xml_str, parser=etree.XMLParser(remove_blank_text=True))
    worldbody = mjcf.find('worldbody')
    # make sphere
    make_source_geom(worldbody, 'food', x=x, y=y, rgba='0 1 0 1', radius=0.5)
    return mjcf


def chemotaxis(xml_str, x, y):
    mjcf = _make_model(xml_str=xml_str, x=x, y=y)
    return etree.tostring(mjcf, pretty_print=True)
