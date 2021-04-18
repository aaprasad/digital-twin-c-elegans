""" OpenAI Gym Swimmer-v3 with specific `n_bodies` and `body_len` to mimic C. elegans nematode """

from .swimmer_gym_v3_v1 import swimmer as swimmer_base
from lxml import etree


def _list_to_str(a):
    return ' '.join([str(f) for f in a])


def _make_model(n_bodies, body_len, xml_file, camera_pos=None):
    """ Generates an xml string defining a swimmer with `n_bodies` bodies and certain `body_len`
    Args:
        n_bodies: number of bodies, >= 3
        body_len: length of body, >= 0.2 (2 times the size of joint)
        xml_file: template xml file path
    """
    # minimum `body_len`
    if body_len < 0.2:
        raise ValueError('`body_len` must be at least two times the size of joint. Received {}'.format(body_len))
    # create swimmer with specific `n_bodies`
    xml_str = swimmer_base(n_bodies=n_bodies, xml_file=xml_file, camera_pos=camera_pos)
    # create mjcf
    mjcf = etree.fromstring(xml_str, parser=etree.XMLParser(remove_blank_text=True))
    # the highest level of body
    body = mjcf.find('worldbody/body')
    fromto = _list_to_str([0, 0, 0, -body_len, 0, 0])
    pos = _list_to_str([-body_len, 0, 0])
    for i in range(n_bodies):
        geom = body.find('geom')
        geom.set('fromto', fromto)
        body = body.find('body')
        if body is not None:
            body.set('pos', pos)
    # get mjcf xml string
    return etree.tostring(mjcf, pretty_print=True)


def swimmer(n_bodies, body_len, xml_file, camera_pos=None):
    return _make_model(n_bodies=n_bodies, body_len=body_len, xml_file=xml_file, camera_pos=camera_pos)
