import gym
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.ellipsoid_v1 import swimmer
from gym_worm.envs.mujoco.joint import remove_joint
from gym_worm.envs.mujoco.position import position
from gym_worm.envs.mujoco.position_actuator import position_actuator
from gym_worm.wrappers.recorder import Recorder
from gym_worm.wrappers.sensor_observation import SensorObservation
from gym_worm.wrappers.skip_frame import SkipFrame


def make_swimmer(n_bodies, joint_range, max_episode_steps, reset_noise_scale):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range)
    xml_str = position(xml_str)
    xml_str = camera(xml_str)
    env = gym.make(
        'Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env


def make_swimmer_with_servo(n_bodies, joint_range, max_episode_steps, reset_noise_scale, kp=1, skip=10):
    """ z-axis and y-axis position servos
    action space: Box(-100.0, 100.0, (48,), float32)
        [0:24]: angles applied on the position servos around z-axis (angle, rad)
        [24:48]: angles applied on the position servos around y-axis (angle, rad)
    """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range)
    xml_str = position_actuator(xml_str, joint_range, kp)
    xml_str = position(xml_str)
    xml_str = camera(xml_str)
    # with open('swimmer.xml', 'w') as f:
    #     f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale
    )
    env = SkipFrame(env, skip)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env


def make_swimmer_y(n_bodies, joint_range, max_episode_steps, reset_noise_scale):
    """ no z-axis rotation dof, only y-axis rotation dof, use motor actuator on y-axis rotation """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range)
    xml_str = remove_joint(xml_str, n_bodies)
    xml_str = position(xml_str)
    xml_str = camera(xml_str)
    # with open('swimmer.xml', 'w') as f:
    #     f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env
