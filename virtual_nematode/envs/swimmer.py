import gym
from gym_worm.envs.mujoco.swimmer_v3_v1 import swimmer
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.friction import friction
from gym_worm.envs.mujoco.position import position
from gym_worm.wrappers.recorder import Recorder
from gym_worm.wrappers.sensor_observation import SensorObservation


def make_swimmer(n_bodies=12, joint_range='-100 100', body_len=0.25, max_episode_steps=1000, reset_noise_scale=0.1):
    """ create swimmer env """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='0 -5 5', camera_z=None)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env


def make_swimmer_friction(n_bodies=12, joint_range='-100 100', body_len=0.25, max_episode_steps=1000, reset_noise_scale=0.1):
    """ create swimmer env """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='0 -5 5', camera_z=None)
    xml_str = friction(xml_str, n_section=7, width=1.5, nconmax=400)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env
