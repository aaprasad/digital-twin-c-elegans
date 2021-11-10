import gym
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.muscle_worm_v0 import swimmer
from gym_worm.envs.mujoco.position import position
from gym_worm.envs.mujoco.tendon import tendon
from gym_worm.wrappers.sensor_observation import SensorObservation


def make_swimmer(
    n_bodies=25, joint_range='-40 40', body_len=0.1, muscle_len=0.1, y_sidesite=0.1, z_medial=0.02, z_lateral=0.04,
    max_episode_steps=1000, reset_noise_scale=0.1
):
    """ create swimmer env """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len, muscle_len, y_sidesite, z_medial, z_lateral, arrangement=None)
    xml_str = tendon(xml_str)
    xml_str = position(xml_str)
    xml_str = camera(xml_str)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    return env
