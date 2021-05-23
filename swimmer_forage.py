""" swimmer: forage
- wrap env with forage task
"""

from src.envs.mujoco.swimmer_gym_v3_v2 import swimmer
from src.envs.mujoco.forage import forage
import gym
import os


def make_swimmer(n_bodies=12, joint_range='-40 40', body_len=0.25, perimeter_width=6, camera_pos='0 -6 6', max_episode_steps=1000):
    """ create swimmer env """
    # generate xml str
    xml_folder = 'src/envs/mujoco/assets/'
    xml_str = swimmer(n_bodies=n_bodies, joint_range=joint_range, body_len=body_len, xml_file=os.path.join(xml_folder, 'swimmer.xml'), camera_pos=camera_pos)
    xml_str = forage(xml_str=xml_str, perimeter_width=perimeter_width)
    # write temp xml file, make env and delete temp file
    xml_file = os.path.join(xml_folder, 'swimmer_temp.xml')
    with open(xml_file, 'wb') as f:
        f.write(xml_str)
    env = gym.make('Swimmer-v3', xml_file=os.path.join(os.getcwd(), xml_file))
    env._max_episode_steps = max_episode_steps
    if os.path.exists(xml_file):
        os.remove(xml_file)
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
            print("Episode finished after {} steps".format(i + 1))
            break
    env.close()


if __name__ == '__main__':
    test_random()
