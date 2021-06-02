""" OpenAI Gym Swimmer-v3 with specific `n_bodies`, `joint_range` and `body_len` """

from .swimmer_gym_v3_v1 import make_model as make_model_base
from lxml import etree


def _list_to_str(a):
    return ' '.join([str(f) for f in a])


def make_model(n_bodies, joint_range, body_len, xml_file, camera_pos=None, camera_z=None):
    """ Generates an xml string defining a swimmer with `n_bodies` bodies and certain `body_len`
    Args:
        n_bodies: number of bodies, >= 3
        body_len: length of body, >= 0.2 (2 times the size of joint)
        xml_file: template xml file path
    """
    # minimum `body_len`
    if body_len < 0.2:
        raise ValueError('`body_len` must be at least two times the size of joint. Received {}'.format(body_len))
    # create swimmer
    mjcf = make_model_base(n_bodies=n_bodies, joint_range=joint_range, xml_file=xml_file, camera_pos=camera_pos, camera_z=camera_z)
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
    return mjcf


def swimmer(n_bodies, joint_range, body_len, xml_file, camera_pos=None, camera_z=None):
    mjcf = make_model(n_bodies=n_bodies, joint_range=joint_range, body_len=body_len, xml_file=xml_file, camera_pos=camera_pos, camera_z=camera_z)
    return etree.tostring(mjcf, pretty_print=True)
