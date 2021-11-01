import gym
from gym_worm.envs.mujoco.swimmer_v3_v1 import swimmer
from gym_worm.envs.mujoco.camera import camera
from gym_worm.wrappers.position import Position
from gym_worm.wrappers.recorder import Recorder
import os


def make_swimmer(
    n_bodies=12, joint_range='-100 100', body_len=0.25, camera_pos='0 -6 6', camera_z=None, camera_name=None,
    max_episode_steps=1000, video_name='swimmer', reset_noise_scale=0.1
):
    """ create swimmer env """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str = camera(xml_str, camera_pos, camera_z)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = Position(env)
    env = Recorder(env, stats_name=['com'], camera_name=camera_name)
    if camera_name is not None:
        env = gym.wrappers.Monitor(env, directory=os.path.join('video', video_name), force=True)
    return env
