""" OpenAI Gym Swimmer-v3 with specific `n_bodies` and `joint_range` """

from lxml import etree


def parse_xml(xml_str):
    """ parse xml and clean it up """
    # create mjcf
    mjcf = etree.fromstring(xml_str, parser=etree.XMLParser(remove_blank_text=True))
    # use unified coordinates: geom fromto, body pos
    geom = mjcf.find('worldbody/body/geom')
    geom.set('fromto', '0 0 0 -1 0 0')
    body = mjcf.find('worldbody/body/body')
    body.set('pos', '-1 0 0')
    # rename body 1
    body = mjcf.find('worldbody/body')
    body.set('name', 'torso1')
    for joint in body.findall('joint'):
        if joint.get('name') == 'rot':
            joint.set('name', 'rot1')
    # rename body 2
    body = body.find('body')
    body.set('name', 'torso2')
    # rename body 3
    body = body.find('body')
    body.set('name', 'torso3')
    return mjcf


def _make_body(body_str, body_idx):
    body = etree.fromstring(body_str)  # add to the back part
    body.set('name', 'torso{}'.format(body_idx))
    joint = body.find('joint')
    joint.set('name', 'rot{}'.format(body_idx))
    return body


def _set_joint_range(mjcf, n_bodies, joint_range: str):
    """ set joint range """
    body = mjcf.find('worldbody/body/body')
    for i in range(1, n_bodies):
        joint = body.find('joint')
        joint.set('range', joint_range)
        body = body.find('body')
    return mjcf


def make_model(n_bodies, joint_range, xml_file, camera_pos=None):
    """ Generates an xml string defining a swimmer with `n_bodies` bodies.
    Args:
        n_bodies: number of bodies, >= 3
        joint_range: range of joint, original setting '-100 100'
        xml_file: template xml file path
    Reference:
        https://github.com/deepmind/dm_control/blob/master/dm_control/suite/swimmer.py
    """
    # minimum `n_bodies`
    if n_bodies < 3:
        raise ValueError('At least 3 bodies required. Received {}'.format(n_bodies))
    # read template xml
    with open(xml_file, 'rb') as f:
        xml_str = f.read()
    # parse xml
    mjcf = parse_xml(xml_str=xml_str)
    # modify mjcf
    if n_bodies > 3:
        # add more bodies
        body_posterior = mjcf.find('worldbody/body/body/body')
        # a str copy of the back part
        body_str = etree.tostring(body_posterior)
        bodies = _make_body(body_str=body_str, body_idx=n_bodies)
        for i in range(n_bodies - 1, 3, -1):
            temp_body = _make_body(body_str=body_str, body_idx=i)
            temp_body.append(bodies)
            bodies = temp_body
        body_posterior.append(bodies)
        # add more motor actuators
        actuator = mjcf.find('actuator')
        for i in range(4, n_bodies + 1):
            motor = etree.Element('motor', attrib={'ctrllimited': 'true', 'ctrlrange': '-1 1', 'gear': '150.0', 'joint': 'rot{}'.format(i)})
            actuator.append(motor)
    # set joint range
    mjcf = _set_joint_range(mjcf=mjcf, n_bodies=n_bodies, joint_range=joint_range)
    # modify camera pos, default: "0 -3 3"
    if camera_pos is not None:
        camera = mjcf.find('worldbody/body/camera')
        camera.set('pos', camera_pos)
    return mjcf


def swimmer(n_bodies, joint_range, xml_file, camera_pos=None):
    mjcf = make_model(n_bodies=n_bodies, joint_range=joint_range, xml_file=xml_file, camera_pos=camera_pos)
    return etree.tostring(mjcf, pretty_print=True)
