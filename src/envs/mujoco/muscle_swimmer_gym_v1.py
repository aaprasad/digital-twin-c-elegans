""" muscle swimmer implemented based on OpenAI Gym Swimmer-v3 with specific `n_bodies` """

from .swimmer_gym_v3_v1 import make_model as make_model_base
from .muscle_swimmer_gym_v0 import transform_into_muscle_model
from lxml import etree


def _make_model(n_bodies, xml_file, camera_pos=None):
    """ Generates an xml string defining a muscle swimmer with specific `n_bodies`.
    Args:
        n_bodies: number of bodies, >= 3
        xml_file: template xml file path
    """
    # create swimmer with specific `n_bodies`
    mjcf = make_model_base(n_bodies=n_bodies, xml_file=xml_file, camera_pos=camera_pos)
    # muscle model
    mjcf = transform_into_muscle_model(mjcf=mjcf, n_bodies=n_bodies)
    return mjcf


def swimmer(n_bodies, xml_file, camera_pos=None):
    mjcf = _make_model(n_bodies=n_bodies, xml_file=xml_file, camera_pos=camera_pos)
    return etree.tostring(mjcf, pretty_print=True)
