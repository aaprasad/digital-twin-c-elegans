""" nematode with accurate muscle arrangement """

from .swimmer_gym_v3_v2 import make_model as make_model_base
from .muscle_swimmer_gym_v0 import prepare_muscle_model, make_geom, make_muscle
from lxml import etree
import math

""" default staggered muscle arrangement with 25 joints (26 bodies)
Citations:
    - https://wormatlas.org/hermaphrodite/musclesomatic/MusSomaticframeset.html
        MusFIG 15A: Arrangement of body wall muscle cells in the adult hermaphrodite.
    - Genetics of cell and axon migrations in Caenorhabditis elegans
        Fig.2
    - Whole-animal connectomes of both Caenorhabditis elegans sexes
        Supplementary Figure
        https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1352-7/MediaObjects/41586_2019_1352_MOESM4_ESM.pdf
"""
ARRANGEMENT = {
    'dorsal_left': {
        'medial': ['dbwml{}'.format(i) if i % 2 == 1 else None for i in range(1, 25)] + [None],
        'lateral': ['dbwml{}'.format(i) if i % 2 == 0 else None for i in range(1, 25)] + [None]
    },
    'dorsal_right': {
        'medial': ['dbwmr{}'.format(i) if i % 2 == 1 else None for i in range(1, 25)] + [None],
        'lateral': ['dbwmr{}'.format(i) if i % 2 == 0 else None for i in range(1, 25)] + [None]
    },
    'ventral_left': {
        'medial': ['vbwml{}'.format(i) if i % 2 == 1 else None for i in range(1, 22)] + [None, 'vbwml22', None, 'vbwml23'],
        'lateral': ['vbwml{}'.format(i) if i % 2 == 0 else None for i in range(1, 22)] + [None, None, None, None]
    },
    'ventral_right': {
        'medial': ['vbwmr{}'.format(i) if i % 2 == 1 else None for i in range(1, 24)] + [None, 'vbwmr24'],
        'lateral': ['vbwmr{}'.format(i) if i % 2 == 0 else None for i in range(1, 24)] + [None, None]
    }
}


def _calculate_site_pos(quadrant, row):
    """ calculate site pos (y, z) for each row in muscle quadrants  """
    if row == 'medial':
        z = 0.02
    else:  # 'lateral'
        z = 0.04
    y = math.sqrt(0.1 ** 2 - z ** 2)
    if quadrant == 'dorsal_left' or quadrant == 'ventral_left':
        z *= -1
    if quadrant == 'ventral_left' or quadrant == 'ventral_right':
        y *= -1
    return y, z


def _arrange_muscle(mjcf, n_bodies, body_len, arrangement):
    """ arrange muscle """
    # prepare muscle model
    mjcf, tendon, actuator = prepare_muscle_model(mjcf=mjcf)
    # arrange muscle
    anterior_body = mjcf.find('worldbody/body')
    for i in range(1, n_bodies):
        posterior_body = anterior_body.find('body')
        # add muscles
        geom = make_geom(body=posterior_body, name='geom{}'.format(i + 1))
        for quadrant in arrangement:
            for row in arrangement[quadrant]:
                muscle_name = arrangement[quadrant][row][i - 1]
                if muscle_name is not None:
                    y, z = _calculate_site_pos(quadrant, row)
                    make_muscle(
                        anterior_body, posterior_body, geom, tendon, actuator, body_len, y=y, z=z,
                        name={
                            'anterior_site': 'site_anterior_{}'.format(muscle_name),
                            'posterior_site': 'site_posterior_{}'.format(muscle_name),
                            'sidesite': 'sidesite_{}'.format(muscle_name),
                            'spatial': 'tendon_{}'.format(muscle_name),
                            'muscle': 'muscle_{}'.format(muscle_name)
                        }
                    )
        # update pointer
        anterior_body = posterior_body
    return mjcf


def _make_model(body_len, xml_file, arrangement=None, camera_pos=None):
    """ Generates an xml string defining a nematode with specific `n_bodies`, `body_len` and muscle arrangement
    Args:
        n_bodies: number of bodies, >= 3
        body_len: length of body, >= 0.2 (2 times the size of joint)
        xml_file: template xml file path
    """
    # get muscle arrangement
    if arrangement is None:
        arrangement = ARRANGEMENT
    # check arrangement format
    for quadrant in arrangement:
        if quadrant not in ARRANGEMENT:
            raise AssertionError('Unidentified key as quadrant {}.'.format(quadrant))
        for row in arrangement[quadrant]:
            if row not in ARRANGEMENT[quadrant]:
                raise AssertionError('Unidentified key as row {}.'.format(row))
    # get `n_bodies`
    n_bodies = len(arrangement['dorsal_left']['medial']) + 1
    # check `n_bodies`
    for quadrant in arrangement:
        for row in arrangement[quadrant]:
            if len(arrangement[quadrant][row]) + 1 != n_bodies:
                raise ValueError('The length of muscle arrangement is not consistent. Row {} {} is {}.'.format(quadrant, row, len(arrangement[quadrant][row])))
    # create swimmer with specific `n_bodies` and `body_len`
    mjcf = make_model_base(n_bodies=n_bodies, body_len=body_len, xml_file=xml_file, camera_pos=camera_pos)
    # muscle model
    mjcf = _arrange_muscle(mjcf=mjcf, n_bodies=n_bodies, body_len=body_len, arrangement=arrangement)
    return mjcf


def nematode(body_len, xml_file, arrangement=None, camera_pos=None):
    mjcf = _make_model(body_len=body_len, xml_file=xml_file, arrangement=arrangement, camera_pos=camera_pos)
    return etree.tostring(mjcf, pretty_print=True)
