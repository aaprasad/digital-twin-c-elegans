""" swimmer: maze
- wrap env with maze task
"""

from src.envs.mujoco.swimmer_gym_v3_v0 import SwimmerEnv
from src.envs.mujoco.swimmer_gym_v3_v2 import swimmer
from src.envs.mujoco.maze import maze
import gym


def make_swimmer(n_bodies=12, joint_range='-100 100', body_len=0.25, perimeter_width=6, box_width=0.5, camera_pos='0 -6 6', camera_z=None, max_episode_steps=1000):
    """ create swimmer env """
    xml_str = swimmer(n_bodies=n_bodies, joint_range=joint_range, body_len=body_len, xml_file='src/envs/mujoco/assets/swimmer.xml', camera_pos=camera_pos, camera_z=camera_z)
    xml_str = maze(xml_str=xml_str, perimeter_width=perimeter_width, box_width=box_width)
    env = SwimmerEnv(xml_str=xml_str.decode('utf-8'))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    return env


def test_random():
    """ take random actions """
    # swimmer multibody model: set up n_bodies, joint_range and body_len
    env = make_swimmer()
    observation = env.reset()
    for i in range(10 ** 6):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} steps'.format(i + 1))
            break
    env.close()


if __name__ == '__main__':
    test_random()
