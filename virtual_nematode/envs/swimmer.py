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


def fick(target, source, sigma=5):
    r = np.linalg.norm(target - source)
    c = np.exp(-r ** 2 / (2 * sigma ** 2))
    return c


def make_chemotaxis_swimmer(seed, trial, distance, n_bodies=12, joint_range='-100 100', body_len=0.25, max_episode_steps=1000, reset_noise_scale=0.1):
    np.random.seed(seed)
    xml_str_base = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str_base = position(xml_str_base)
    xml_str_base = camera(xml_str_base)
    envs = []
    for _ in range(trial):
        angle = np.random.uniform(0, 2 * np.pi)
        x, y = distance * np.cos(angle), distance * np.sin(angle)
        xml_str = chemotaxis(copy.deepcopy(xml_str_base), x, y)
        env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
        env = gym.wrappers.ClipAction(env)
        env = SensorObservation(env)
        env = DistributionObservation(env, dt=env.dt, f=fick, source=[x, y])
        env = Recorder(env, camera_name=None)
        envs.append(env)
    return envs
