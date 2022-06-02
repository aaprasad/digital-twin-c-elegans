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
import numpy as np
from virtual_nematode.envs.swimmer import fick


def make_swimmer(n_bodies, joint_range, max_episode_steps, reset_noise_scale, density, viscosity, condim, friction):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction)
    xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorpos')
    xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorvel')
    xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorfrc')
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    # with open('swimmer.xml', 'w') as f:
    #     f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Worm-v0', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env


def make_chemotaxis_swimmer(return_func, angle, xml_str_base, distance, reset_noise_scale, max_episode_steps, position_func, camera_name):
    """ create chemotaxis swimmer env """
    def func():
        x, y = distance * np.cos(angle), distance * np.sin(angle)
        xml_str = chemotaxis(copy.deepcopy(xml_str_base), x, y)
        env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale, exclude_current_positions_from_observation=False)
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
        env = gym.wrappers.ClipAction(env)
        env = SensorObservation(env)
        env = DistributionObservation(env, dt=env.dt, f=fick, source=[x, y], position_func=position_func)
        if camera_name is not None:
            env = Recorder(env, camera_name=camera_name)
            env = gym.wrappers.Monitor(env, directory='video/swimmer_weathervane', force=True)
        return env
    if return_func is False:
        return func()
    else:
        return func


def make_chemotaxis_swimmers(
    seed, trials, distance, position_func, n_bodies=12, joint_range='-100 100', max_episode_steps=1000,
    reset_noise_scale=0.1, camera_name=None, return_func=False
):
    """ create a list of chemotaxis swimmer envs """
    np.random.seed(seed)
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density=4000, viscosity=0.1, condim=3, friction='0.04 0.4')
    xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorpos')
    xml_str = actuator(xml_str, n_bodies, sensor_type='actuatorvel')
    xml_str_base = actuator(xml_str, n_bodies, sensor_type='actuatorfrc')
    xml_str_base = position(xml_str_base)
    xml_str_base = camera(xml_str_base)
    envs = []
    for _ in range(trials):
        angle = np.random.uniform(0, 2 * np.pi)
        env = make_chemotaxis_swimmer(return_func, angle, xml_str_base, distance, reset_noise_scale, max_episode_steps, position_func, camera_name)
        envs.append(env)
    return envs
