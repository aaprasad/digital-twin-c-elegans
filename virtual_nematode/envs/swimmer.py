import copy
import gym
from gym_worm.envs.mujoco.chemotaxis import chemotaxis
from gym_worm.envs.mujoco.swimmer_v3_v1 import swimmer
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.position import position
from gym_worm.wrappers.distribution_observation import DistributionObservation
from gym_worm.wrappers.recorder import Recorder
from gym_worm.wrappers.sensor_observation import SensorObservation
import numpy as np


def make_swimmer(n_bodies=12, joint_range='-100 100', body_len=0.25, max_episode_steps=1000, reset_noise_scale=0.1):
    """ create swimmer env """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str = position(xml_str)
    xml_str = camera(xml_str)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env


def fick(target=None, source=None, sigma=5):
    """ gaussian concentration distribution in (0, 1]
    c = e^{-r^2/(2*sigma^2)}
    dc/dr = -r/(sigma^2)*e^{-r^2/(2*sigma^2)}
        odd function
        max(abs(dc/dr))=e^(-0.5)/sigma, where r = sigma or r = -sigma
    d^2c/dr^2 = (r^2-sigma^2)/(sigma^4)*e^{-r^2/(2*sigma^2)}
        d^2c/dr^2 = 0 -> r = sigma or r = -sigma, where dc/dr reaches max abs value
    """
    if target is None or source is None:
        return np.exp(-0.5) / sigma  # max abs gradient
    r = np.linalg.norm(target - source)  # L2 distance
    c = np.exp(-r ** 2 / (2 * sigma ** 2))  # concentration
    return c


def fick_uniform(target, source, concentration=0):
    return concentration


def make_chemotaxis_swimmer(return_func, angle, xml_str_base, distance, reset_noise_scale, max_episode_steps, position_func, camera_name):
    """ create chemotaxis swimmer env """
    def func():
        x, y = distance * np.cos(angle), distance * np.sin(angle)
        xml_str = chemotaxis(copy.deepcopy(xml_str_base), x, y)
        env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
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
    seed, trials, distance, position_func, n_bodies=12, joint_range='-100 100', body_len=0.25, max_episode_steps=1000,
    reset_noise_scale=0.1, camera_name=None, return_func=False
):
    """ create a list of chemotaxis swimmer envs """
    np.random.seed(seed)
    xml_str_base = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str_base = position(xml_str_base)
    xml_str_base = camera(xml_str_base)
    envs = []
    for _ in range(trials):
        angle = np.random.uniform(0, 2 * np.pi)
        env = make_chemotaxis_swimmer(return_func, angle, xml_str_base, distance, reset_noise_scale, max_episode_steps, position_func, camera_name)
        envs.append(env)
    return envs
