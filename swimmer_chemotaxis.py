""" swimmer: chemotaxis """

import gym
import numpy as np
import os
from src.envs.mujoco.swimmer_gym_v3_v2 import swimmer
from src.envs.mujoco.chemtoaxis import chemotaxis


def make_swimmer(n_bodies, joint_range, body_len, camera_pos, max_episode_steps, x, y):
    """ create swimmer env """
    # generate xml str
    xml_folder = 'src/envs/mujoco/assets/'
    xml_str = swimmer(n_bodies=n_bodies, joint_range=joint_range, body_len=body_len, xml_file=os.path.join(xml_folder, 'swimmer.xml'), camera_pos=camera_pos)
    xml_str = chemotaxis(xml_str=xml_str, x=x, y=y)
    # write temp xml file, make env and delete temp file
    xml_file = os.path.join(xml_folder, 'swimmer_temp.xml')
    with open(xml_file, 'wb') as f:
        f.write(xml_str)
    env = gym.make('Swimmer-v3', xml_file=os.path.join(os.getcwd(), xml_file))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if os.path.exists(xml_file):
        os.remove(xml_file)
    return env


def test_random():
    """ take random actions
    multibody model:
        - radius=0.04mm, body_len=0.1mm, n_bodies=12, q_max=0.69rad (~39.53409 degrees)
        - joint_size=0.1 (radius) -> body_len=0.25
        - citation: A computational model of internal representations of chemical gradients in environments for chemotaxis of Caenorhabditis elegans
    """
    x = np.random.randint(-20, 20)
    y = np.sqrt(20 ** 2 - x ** 2)
    env = make_swimmer(n_bodies=12, joint_range='-40 40', body_len=0.25, camera_pos='0 -6 6', max_episode_steps=1500, x=x, y=y)
    observation = env.reset()
    for i in range(10 ** 6):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    env.close()


if __name__ == '__main__':
    test_random()
