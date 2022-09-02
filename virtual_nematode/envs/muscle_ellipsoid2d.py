import copy
import gym
from gym_worm.envs.mujoco.chemotaxis import chemotaxis
from gym_worm.envs.mujoco.sensor import actuator
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.muscle_ellipsoid2d_v0 import swimmer
from gym_worm.envs.mujoco.position import position
from gym_worm.wrappers.distribution_observation import DistributionObservation
from gym_worm.wrappers.recorder import Recorder
from gym_worm.wrappers.sensor_observation import SensorObservation
from virtual_nematode.envs.swimmer import fick


def make_swimmer(n_bodies, joint_range, max_episode_steps, reset_noise_scale, density, viscosity, condim, friction):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction)
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


def make_swimmer_weathervane(
    n_bodies, joint_range, max_episode_steps, reset_noise_scale, distance, position_func, density, viscosity, condim,
    friction, source=(0, 0)
):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction)
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
    env = DistributionObservation(env, dt=env.dt, f=fick, source=source, position_func=position_func)
    env = Recorder(env, camera_name=None)
    return env


def make_chemotaxis_swimmer(return_func, source, xml_str_base, reset_noise_scale, max_episode_steps, position_func):
    """ create chemotaxis swimmer env """
    def func():
        x, y = source
        xml_str = chemotaxis(copy.deepcopy(xml_str_base), x, y)
        # with open('swimmer.xml', 'w') as f:
        #     f.write(xml_str.decode('utf-8'))
        env = gym.make(
            'Swimmer-v3-v1', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
            reset_noise_scale=reset_noise_scale
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
        env = gym.wrappers.ClipAction(env)
        env = SensorObservation(env)
        env = DistributionObservation(env, dt=env.dt, f=fick, source=source, position_func=position_func)
        env = Recorder(env, camera_name=None)
        return env
    if return_func is False:
        return func()
    else:
        return func


def make_chemotaxis_swimmers(
    sources, position_func, n_bodies, joint_range, max_episode_steps, reset_noise_scale, density,
    viscosity, condim, friction, return_func=False
):
    """ create a list of chemotaxis swimmer envs """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction)
    # xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorpos')
    # xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorvel')
    # xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorfrc')
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    envs = []
    for source in sources:
        env = make_chemotaxis_swimmer(return_func, source, xml_str, reset_noise_scale, max_episode_steps, position_func)
        envs.append(env)
    return envs
