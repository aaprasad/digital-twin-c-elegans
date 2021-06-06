""" muscle swimmer with specific `n_bodies`, `joint_range`, `body_len` and `muscle_len` """

from src.envs.mujoco.swimmer_gym_v3_v0 import SwimmerEnv
from src.envs.mujoco.muscle_swimmer_gym_v2 import swimmer
import gym


def make_swimmer(n_bodies, joint_range, body_len, muscle_len, camera_pos, camera_z):
    """ create swimmer env """
    xml_str = swimmer(n_bodies=n_bodies, joint_range=joint_range, body_len=body_len, muscle_len=muscle_len, xml_file='src/envs/mujoco/assets/swimmer.xml', camera_pos=camera_pos, camera_z=camera_z)
    env = SwimmerEnv(xml_str=xml_str.decode('utf-8'))
    print(env.action_space, env.action_space.low, env.action_space.high)
    print(env.observation_space, env.observation_space.low, env.observation_space.high)
    return env


def test_random():
    """ take random actions with gym env """
    # bode_len >= 0.2
    env = make_swimmer(n_bodies=5, joint_range='-100 100', body_len=0.5, muscle_len=0.26, camera_pos='0 -5 5', camera_z=None)
    # record video
    env = gym.wrappers.Monitor(env, directory='video/muscle_swimmer_gym_v2', force=True)
    # run and record video
    observation = env.reset()
    for i in range(1000):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    env.close()


if __name__ == '__main__':
    test_random()
