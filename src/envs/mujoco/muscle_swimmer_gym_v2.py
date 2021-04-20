""" muscle swimmer implemented based on OpenAI Gym Swimmer-v3 with specific `n_bodies` and `body_len` """

from .swimmer_gym_v3_v2 import make_model as make_model_base
from .muscle_swimmer_gym_v0 import arrange_muscle
from lxml import etree


def _make_model(n_bodies, body_len, xml_file, camera_pos=None):
    """ Generates an xml string defining a swimmer with `n_bodies` bodies and certain `body_len`
    Args:
        n_bodies: number of bodies, >= 3
        body_len: length of body, >= 0.2 (2 times the size of joint)
        xml_file: template xml file path
    """
    # create swimmer with specific `n_bodies`
    mjcf = make_model_base(n_bodies=n_bodies, body_len=body_len, xml_file=xml_file, camera_pos=camera_pos)
    # muscle model
    mjcf = arrange_muscle(mjcf=mjcf, n_bodies=n_bodies, body_len=body_len)
    return mjcf


def swimmer(n_bodies, body_len, xml_file, camera_pos=None):
    mjcf = _make_model(n_bodies=n_bodies, body_len=body_len, xml_file=xml_file, camera_pos=camera_pos)
    return etree.tostring(mjcf, pretty_print=True)
