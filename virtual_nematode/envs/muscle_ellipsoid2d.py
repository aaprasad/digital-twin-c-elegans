import gym
from gym_worm.envs.mujoco.chemotaxis import chemotaxis
from gym_worm.envs.mujoco.sensor import actuator
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.muscle_ellipsoid2d_v0 import swimmer
from gym_worm.envs.mujoco.position import position
from gym_worm.envs.mujoco.trapped import trapped
from gym_worm.wrappers.distribution_observation import DistributionObservation
from gym_worm.wrappers.muscle_action import get_action_index, PartialAction
from gym_worm.wrappers.recorder import Recorder
from gym_worm.wrappers.sensor_observation import SensorObservation
from virtual_nematode.envs.swimmer import fick


def make_swimmer(n_bodies, joint_range, max_episode_steps, reset_noise_scale, density, viscosity, condim, friction, cone):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction, cone)
    # xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorpos')
    # xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorvel')
    # xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorfrc')
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    # with open('swimmer.xml', 'w') as f:
    #     f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v1', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env


def make_swimmer_xml(n_bodies, joint_range, density, viscosity, condim, friction, cone):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction, cone)
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    return xml_str


def make_swimmer_fn(xml_str, max_episode_steps, reset_noise_scale):
    def _fn():
        env = gym.make(
            'Swimmer-v3-v1', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
            reset_noise_scale=reset_noise_scale
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
        env = gym.wrappers.ClipAction(env)
        env = SensorObservation(env)
        env = Recorder(env, camera_name=None)
        return env
    return _fn


def make_swimmer_trapped(
    n_bodies, joint_range, max_episode_steps, reset_noise_scale, density, viscosity, condim, friction, body_index,
    muscle_index
):
    """ swimmer with trapped body segments
    body_index: list of body index within [1, 2, ..., 25]
    muscle_index: list of muscle index within [1, 2, ..., 24]
    """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction)
    xml_str = trapped(xml_str, muscle_index, body_index)
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    # with open('swimmer.xml', 'w') as f:
    #     f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v1', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = PartialAction(env, action_index=get_action_index(muscle_index))
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env


def make_swimmer_weathervane(
    n_bodies, joint_range, max_episode_steps, reset_noise_scale, distance, source, position_func, density,
    viscosity, condim, friction, cone, distribution_func=fick
):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction, cone)
    # xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorpos')
    # xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorvel')
    # xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorfrc')
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    xml_str = chemotaxis(xml_str, x=source[0], y=source[1])
    # with open('swimmer.xml', 'w') as f:
    #     f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v2', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale, distance=distance
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = DistributionObservation(env, dt=env.dt, f=distribution_func, source=source, position_func=position_func)
    env = Recorder(env, camera_name=None)
    return env


def make_swimmer_weathervane_xml(n_bodies, joint_range, source, density, viscosity, condim, friction, cone):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction, cone)
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    xml_str = chemotaxis(xml_str, x=source[0], y=source[1])
    return xml_str


def make_swimmer_weathervane_fn(xml_str, reset_noise_scale, distance, max_episode_steps, source, position_func, distribution_func=fick):
    def _fn():
        env = gym.make(
            'Swimmer-v3-v2', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
            reset_noise_scale=reset_noise_scale, distance=distance
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
        env = gym.wrappers.ClipAction(env)
        env = SensorObservation(env)
        env = DistributionObservation(env, dt=env.dt, f=distribution_func, source=source, position_func=position_func)
        env = Recorder(env, camera_name=None)
        return env
    return _fn


def make_swimmer_weathervane_fixed(
    n_bodies, joint_range, max_episode_steps, reset_noise_scale, pos, source, position_func, density,
    viscosity, condim, friction, cone
):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction, cone)
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    xml_str = chemotaxis(xml_str, x=source[0], y=source[1])
    # with open('swimmer.xml', 'w') as f:
    #     f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v1', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale, position=pos
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = DistributionObservation(env, dt=env.dt, f=fick, source=source, position_func=position_func)
    env = Recorder(env, camera_name=None)
    return env
