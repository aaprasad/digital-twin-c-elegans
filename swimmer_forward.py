""" swimmer: forward movement """

import gym
import numpy as np
import os
from src.envs.mujoco.swimmer_gym_v3_v0 import SwimmerEnv
from src.envs.mujoco.swimmer_gym_v3_v2 import swimmer
from src.models.forward import Forward
from src.wrappers.position import Position
from src.wrappers.recorder import Recorder


def make_swimmer(
    n_bodies=12, joint_range='-100 100', body_len=0.25, camera_pos='0 -6 6', camera_z=None, camera_name=None,
    max_episode_steps=1000, video_name='swimmer_forward', reset_noise_scale=0.1
):
    """ create swimmer env """
    xml_str = swimmer(n_bodies, joint_range, body_len, 'src/envs/mujoco/assets/swimmer.xml', camera_pos, camera_z)
    env = SwimmerEnv(xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = Position(env)
    env = Recorder(env, stats_name=['com'], camera_name=camera_name)
    if camera_name is not None:
        env = gym.wrappers.Monitor(env, directory=os.path.join('video', video_name), force=True)
    return env


def test_forward(seed=None, max_episode_steps=2500):
    """ forward sinusoidal movement """
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=0.1, camera_z=50, camera_name=None)
    env.seed(seed)
    observation = env.reset()
    model = Forward(dt=env.dt, seed=seed)
    for i in range(10 ** 6):
        env.render()
        action = model.step(step=i, q=observation[1:12], q_vel=observation[15:])
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} steps'.format(i + 1))
            break
    displacement = np.linalg.norm(np.array(env.stats['com'][-1]) - np.array(env.stats['com'][0]), ord=2)
    print('com displacement {:.2f} / {} steps'.format(displacement, max_episode_steps))
    env.close()


if __name__ == '__main__':
    test_forward()
