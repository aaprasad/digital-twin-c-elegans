import gym
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.ellipsoid_v1 import swimmer
from gym_worm.envs.mujoco.position import position
from gym_worm.envs.mujoco.position_actuator import position_actuator
from gym_worm.wrappers.recorder import Recorder
from gym_worm.wrappers.sensor_observation import SensorObservation
from gym_worm.wrappers.skip_frame import SkipFrame


def make_swimmer(n_bodies, joint_range, body_len, max_episode_steps, reset_noise_scale, kp=1, skip=10):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str = position_actuator(xml_str, joint_range, kp)
    xml_str = position(xml_str)
    xml_str = camera(xml_str)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = SkipFrame(env, skip)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env
