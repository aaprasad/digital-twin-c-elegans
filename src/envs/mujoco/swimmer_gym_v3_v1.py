""" OpenAI Gym Swimmer-v3 with specific `n_bodies` to mimic C. elegans nematode """

from lxml import etree


def _make_body(body_str, body_idx):
    body = etree.fromstring(body_str)  # add to the back part
    body.attrib['name'] = 'back' + str(body_idx)
    body.find('joint').attrib['name'] = 'rot' + str(body_idx)
    return body


def _make_model(n_bodies, xml_file, camera_pos=None):
    """ Generates an xml string defining a swimmer with `n_bodies` bodies.
    Args:
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
    # create mjcf
    mjcf = etree.fromstring(xml_str)
    # modify mjcf
    if n_bodies > 3:
        # add more bodies
        back = mjcf.find('worldbody/body/body/body')
        # a str copy of the back part
        back_str = etree.tostring(back)
        body = _make_body(body_str=back_str, body_idx=n_bodies)
        for i in range(n_bodies - 1, 3, -1):
            temp_body = _make_body(body_str=back_str, body_idx=i)
            temp_body.append(body)
            body = temp_body
        back.append(body)
        # add more motor actuators
        actuator = mjcf.find('actuator')
        motor = actuator.find('motor')
        motor_str = etree.tostring(motor)
        for i in range(4, n_bodies + 1):
            temp_motor = etree.fromstring(motor_str)
            temp_motor.attrib['joint'] = 'rot' + str(i)
            actuator.append(temp_motor)
    # modify camera pos, default: "0 -3 3"
    if camera_pos is not None:
        camera = mjcf.find('worldbody/body/camera')
        camera.attrib['pos'] = camera_pos
    # get mjcf xml string
    return etree.tostring(mjcf, pretty_print=True)


def swimmer(n_bodies, xml_file, camera_pos=None):
    return _make_model(n_bodies=n_bodies, xml_file=xml_file, camera_pos=camera_pos)
