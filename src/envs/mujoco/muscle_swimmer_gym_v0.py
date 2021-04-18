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


def _make_muscle(index, body, tendon, actuator):
    """ create a pair of muscles at a joint connecting anterior and posterior torso
    Args:
        index: the index of the anterior torso
        body: the anterior torso
    """
    # anterior torso sites
    anterior_torso_dorsal_site = etree.Element(
        'site', attrib={'name': 'torso{}_posterior_dorsal'.format(index), 'pos': '-0.87 0.1 0'}
    )
    anterior_torso_ventral_site = etree.Element(
        'site', attrib={'name': 'torso{}_posterior_ventral'.format(index), 'pos': '-0.87 -0.1 0'}
    )
    body.append(anterior_torso_dorsal_site)
    body.append(anterior_torso_ventral_site)

    # posterior torso
    body = body.find('body')
    # joint geom without mass (the mass of joint is included in torso)
    geom = etree.Element(
        'geom', attrib={
            'name': 'geom{}'.format(index + 1), 'pos': '0 0 0', 'size': '0.1', 'type': 'sphere', 'density': '0',
            'rgba': '0.8 0.2 0.1 1'
        }
    )
    body.append(geom)
    # posterior torso sites
    posterior_torso_dorsal_site = etree.Element(
        'site', attrib={'name': 'torso{}_anterior_dorsal'.format(index + 1), 'pos': '-0.13 0.1 0'}
    )
    posterior_torso_ventral_site = etree.Element(
        'site', attrib={'name': 'torso{}_anterior_ventral'.format(index + 1), 'pos': '-0.13 -0.1 0'}
    )
    body.append(posterior_torso_dorsal_site)
    body.append(posterior_torso_ventral_site)

    # sidesite
    sidesite_dorsal = etree.Element(
        'site', attrib={'name': 'sidesite{}_dorsal'.format(index + 1), 'pos': '0 0.11 0', 'rgba': '0.8 0.2 0.1 0'}
    )
    sidesite_ventral = etree.Element(
        'site', attrib={'name': 'sidesite{}_ventral'.format(index + 1), 'pos': '0 -0.11 0', 'rgba': '0.8 0.2 0.1 0'}
    )
    body.append(sidesite_dorsal)
    body.append(sidesite_ventral)

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


def _make_model(xml_file):
    """ Generates an xml string defining a muscle swimmer.
    Args:
        xml_file: template xml file path
    """
    # parse template xml
    mjcf = parse_xml(xml_file)
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
    # add muscle
    body = mjcf.find('worldbody/body')
    for i in range(1, 3):
        _make_muscle(index=i, body=body, tendon=tendon, actuator=actuator)
        body = body.find('body')

    # get mjcf xml string
    return etree.tostring(mjcf, pretty_print=True)


def swimmer(xml_file):
    return _make_model(xml_file=xml_file)
