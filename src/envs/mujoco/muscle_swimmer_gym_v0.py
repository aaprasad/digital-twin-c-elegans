""" muscle swimmer implemented based on OpenAI Gym Swimmer-v3
- swimmer lies on its left side, with dorsal side on its left (y > 0) and ventral side on its right (y < 0)
"""

from .swimmer_gym_v3_v1 import parse_xml
from lxml import etree


def _make_spatial(anterior_torso_site, posterior_torso_site, sidesite, geom, tendon, name):
    """ create spatial tendon between anterior torso and posterior torso """
    spatial = etree.Element('spatial', attrib={'name': name, 'width': '0.002', 'rgba': '0.95 0.3 0.3 1'})
    spatial.append(etree.Element('pulley', attrib={'divisor': '2'}))
    spatial.append(etree.Element('site', attrib={'site': anterior_torso_site.get('name')}))
    spatial.append(etree.Element('geom', attrib={'geom': geom.get('name'), 'sidesite': sidesite.get('name')}))
    spatial.append(etree.Element('site', attrib={'site': posterior_torso_site.get('name')}))
    tendon.append(spatial)


def prepare_muscle_model(mjcf):
    """ muscle model preliminary set up
    - set up muscle model default settings
    - remove joint motor actuators
    - add empty tendon, actuator elements
    """
    # default site attributions
    default = mjcf.find('default')
    default.append(etree.Element('site', attrib={'type': 'sphere', 'size': '0.01'}))
    # default muscle attributions
    default.append(etree.Element('muscle', attrib={'ctrllimited': 'true', 'ctrlrange': '0 1'}))
    # remove all joints' motor actuators
    mjcf.remove(mjcf.find('actuator'))
    # add tendon element
    tendon = etree.Element('tendon', attrib={})
    mjcf.append(tendon)
    # add actuator element
    actuator = etree.Element('actuator', attrib={})
    mjcf.append(actuator)
    return mjcf, tendon, actuator


def make_geom(body, name):
    """ joint geom without mass (the mass of joint is included in torso)
    - red color, but transparent
    """
    geom = etree.Element('geom', attrib={'name': name, 'pos': '0 0 0', 'size': '0.1', 'type': 'sphere', 'density': '0', 'rgba': '0.8 0.2 0.1 0'})
    body.insert(-1, geom)
    return geom


def make_sidesite(body, side: str, name):
    """ sidesite
    Args:
        side: 'dorsal' or 'ventral'
    """
    if side == 'dorsal':
        sidesite_pos = '0 0.15 0'
    elif side == 'ventral':
        sidesite_pos = '0 -0.15 0'
    else:
        raise ValueError('side is dorsal or ventral. Received {}'.format(side))
    sidesite = etree.Element('site', attrib={'name': name, 'pos': sidesite_pos, 'rgba': '0.8 0.2 0.1 0'})
    body.insert(-1, sidesite)
    return sidesite


def make_muscle(anterior_body, posterior_body, geom, tendon, actuator, body_len, muscle_len, y, z, sidesite, name):
    """ wrap tendon around geom through sidesite and add muscle actuator
    Args:
        geom: geom for tendon wrapping
        muscle_len: the len of muscle is equally split between anterior and posterior bodies
            - muscle_len >= 0.26 when joint_range = '-100 100'
            - muscle_len >= 0.2 (2 * joint size) when joint_range = '-90 90'
        y: y ** 2 + z ** 2 == 0.1 ** 2, -0.1 <= y <= 0.1
        z: the elevation of sites, -0.1 <= z <= 0.1
        name: names of the components
    """
    # anterior torso sites
    anterior_site = etree.Element('site', attrib={'name': name['anterior_site'], 'pos': '{} {} {}'.format(-body_len + muscle_len / 2., y, z)})
    anterior_body.insert(-1, anterior_site)
    # posterior torso sites
    posterior_site = etree.Element('site', attrib={'name': name['posterior_site'], 'pos': '{} {} {}'.format(-muscle_len / 2., y, z)})
    posterior_body.insert(-1, posterior_site)
    # spatial
    _make_spatial(anterior_site, posterior_site, sidesite, geom, tendon, name['spatial'])
    # actuator
    muscle = etree.Element('muscle', attrib={'name': name['muscle'], 'tendon': name['spatial']})
    actuator.append(muscle)


def arrange_muscle(mjcf, n_bodies, body_len, muscle_len):
    """ transform a joint based model into muscle based model """
    # prepare muscle model
    mjcf, tendon, actuator = prepare_muscle_model(mjcf=mjcf)
    # arrange muscle
    z = 0
    y_abs = 0.1  # y = sqrt(0.1 ** 2 - z ** 2)
    anterior_body = mjcf.find('worldbody/body')
    for i in range(1, n_bodies):
        posterior_body = anterior_body.find('body')
        # create a pair of muscles at a joint connecting anterior and posterior torso
        geom = make_geom(body=posterior_body, name='geom{}'.format(i + 1))
        make_muscle(
            anterior_body, posterior_body, geom, tendon, actuator, body_len, muscle_len, y=y_abs, z=z,
            sidesite=make_sidesite(body=posterior_body, side='dorsal', name='sidesite{}_dorsal'.format(i + 1)),
            name={
                'anterior_site': 'torso{}_posterior_dorsal'.format(i),
                'posterior_site': 'torso{}_anterior_dorsal'.format(i + 1),
                'spatial': 'tendon{}_dorsal'.format(i + 1),
                'muscle': 'muscle{}_dorsal'.format(i + 1)
            }
        )
        make_muscle(
            anterior_body, posterior_body, geom, tendon, actuator, body_len, muscle_len, y=-y_abs, z=z,
            sidesite=make_sidesite(body=posterior_body, side='ventral', name='sidesite{}_ventral'.format(i + 1)),
            name={
                'anterior_site': 'torso{}_posterior_ventral'.format(i),
                'posterior_site': 'torso{}_anterior_ventral'.format(i + 1),
                'spatial': 'tendon{}_ventral'.format(i + 1),
                'muscle': 'muscle{}_ventral'.format(i + 1)
            }
        )
        # update pointer
        anterior_body = posterior_body
    return mjcf


def _make_model(xml_file):
    """ Generates an xml string defining a muscle swimmer.
    Args:
        xml_file: template xml file path
    """
    # read template xml
    with open(xml_file, 'rb') as f:
        xml_str = f.read()
    # parse template xml
    mjcf = parse_xml(xml_str=xml_str)
    # muscle model
    mjcf = arrange_muscle(mjcf=mjcf, n_bodies=3, body_len=1, muscle_len=0.26)
    return mjcf


def swimmer(xml_file):
    mjcf = _make_model(xml_file=xml_file)
    return etree.tostring(mjcf, pretty_print=True)
