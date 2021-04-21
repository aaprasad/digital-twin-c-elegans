""" muscle swimmer with specific `n_bodies` and `joint_range` """

from .swimmer_gym_v3_v1 import make_model as make_model_base
from .muscle_swimmer_gym_v0 import arrange_muscle
from lxml import etree


def _make_model(n_bodies, joint_range, xml_file, camera_pos=None):
    """ Generates an xml string defining a muscle swimmer with specific `n_bodies`.
    Args:
        n_bodies: number of bodies, >= 3
        xml_file: template xml file path
    """
    # create swimmer
    mjcf = make_model_base(n_bodies=n_bodies, joint_range=joint_range, xml_file=xml_file, camera_pos=camera_pos)
    # muscle model
    mjcf = arrange_muscle(mjcf=mjcf, n_bodies=n_bodies, body_len=1)
    return mjcf


def swimmer(n_bodies, joint_range, xml_file, camera_pos=None):
    mjcf = _make_model(n_bodies=n_bodies, joint_range=joint_range, xml_file=xml_file, camera_pos=camera_pos)
    return etree.tostring(mjcf, pretty_print=True)
