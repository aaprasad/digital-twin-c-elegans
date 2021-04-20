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


def _make_muscle(index, body, tendon, actuator, body_len):
    """ create a pair of muscles at a joint connecting anterior and posterior torso
    Args:
        index: the index of the anterior torso
        body: the anterior torso
    """
    # anterior torso sites
    anterior_torso_dorsal_site = etree.Element(
        'site', attrib={'name': 'torso{}_posterior_dorsal'.format(index), 'pos': '{} 0.1 0'.format(-body_len + 0.1 + 0.03)}
    )
    anterior_torso_ventral_site = etree.Element(
        'site', attrib={'name': 'torso{}_posterior_ventral'.format(index), 'pos': '{} -0.1 0'.format(-body_len + 0.1 + 0.03)}
    )
    body.insert(-1, anterior_torso_dorsal_site)
    body.insert(-1, anterior_torso_ventral_site)

    # posterior torso
    body = body.find('body')
    # joint geom without mass (the mass of joint is included in torso)
    geom = etree.Element(
        'geom', attrib={
            'name': 'geom{}'.format(index + 1), 'pos': '0 0 0', 'size': '0.1', 'type': 'sphere', 'density': '0',
            'rgba': '0.8 0.2 0.1 1'
        }
    )
    body.insert(-1, geom)
    # posterior torso sites
    posterior_torso_dorsal_site = etree.Element(
        'site', attrib={'name': 'torso{}_anterior_dorsal'.format(index + 1), 'pos': '{} 0.1 0'.format(-0.1 - 0.03)}
    )
    posterior_torso_ventral_site = etree.Element(
        'site', attrib={'name': 'torso{}_anterior_ventral'.format(index + 1), 'pos': '{} -0.1 0'.format(-0.1 - 0.03)}
    )
    body.insert(-1, posterior_torso_dorsal_site)
    body.insert(-1, posterior_torso_ventral_site)

    # sidesite
    sidesite_dorsal = etree.Element(
        'site', attrib={'name': 'sidesite{}_dorsal'.format(index + 1), 'pos': '0 0.11 0', 'rgba': '0.8 0.2 0.1 0'}
    )
    sidesite_ventral = etree.Element(
        'site', attrib={'name': 'sidesite{}_ventral'.format(index + 1), 'pos': '0 -0.11 0', 'rgba': '0.8 0.2 0.1 0'}
    )
    body.insert(-1, sidesite_dorsal)
    body.insert(-1, sidesite_ventral)

    # spatial
    spatial_dorsal_name = 'tendon{}_dorsal'.format(index + 1)
    _make_spatial(anterior_torso_dorsal_site, posterior_torso_dorsal_site, sidesite_dorsal, geom, tendon, spatial_dorsal_name)
    spatial_ventral_name = 'tendon{}_ventral'.format(index + 1)
    _make_spatial(anterior_torso_ventral_site, posterior_torso_ventral_site, sidesite_ventral, geom, tendon, spatial_ventral_name)

    # actuator
    muscle_dorsal = etree.Element('muscle', attrib={'name': 'muscle{}_dorsal'.format(index + 1), 'tendon': spatial_dorsal_name})
    muscle_ventral = etree.Element('muscle', attrib={'name': 'muscle{}_ventral'.format(index + 1), 'tendon': spatial_ventral_name})
    actuator.append(muscle_dorsal)
    actuator.append(muscle_ventral)


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


def arrange_muscle(mjcf, n_bodies, body_len):
    """ transform a joint based model into muscle based model """
    # prepare muscle model
    mjcf, tendon, actuator = prepare_muscle_model(mjcf=mjcf)
    # add muscle
    body = mjcf.find('worldbody/body')
    for i in range(1, n_bodies):
        _make_muscle(index=i, body=body, tendon=tendon, actuator=actuator, body_len=body_len)
        body = body.find('body')
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
    mjcf = arrange_muscle(mjcf=mjcf, n_bodies=3, body_len=1)
    return mjcf


def swimmer(xml_file):
    mjcf = _make_model(xml_file=xml_file)
    return etree.tostring(mjcf, pretty_print=True)
