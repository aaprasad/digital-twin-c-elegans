""" muscle swimmer with specific `n_bodies` and `joint_range` """

from src.envs.mujoco.swimmer_gym_v3_v0 import SwimmerEnv
from src.envs.mujoco.muscle_swimmer_gym_v1 import swimmer
import gym


def make_swimmer(n_bodies, joint_range, camera_pos, camera_z):
    """ create swimmer env """
    xml_str = swimmer(n_bodies=n_bodies, joint_range=joint_range, xml_file='src/envs/mujoco/assets/swimmer.xml', camera_pos=camera_pos, camera_z=camera_z)
    env = SwimmerEnv(xml_str=xml_str.decode('utf-8'))
    # action: Box(0.0, 1.0, (8,), float32)
    print(env.action_space, env.action_space.low, env.action_space.high)
    # observation: Box(-inf, inf, (12,), float64)
    print(env.observation_space, env.observation_space.low, env.observation_space.high)
    return env


def test_random():
    """ take random actions with gym env """
    # make env
    env = make_swimmer(n_bodies=5, joint_range='-100 100', camera_pos='0 -6 6', camera_z=None)
    # record video
    env = gym.wrappers.Monitor(env, directory='video/muscle_swimmer_gym_v1', force=True)
    # run and record video
    observation = env.reset()
    reward_list = []
    for i in range(1000):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        reward_list.append(reward)
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    # n_bodies=5, joint_range='-10 10': ~ -180
    # n_bodies=5, joint_range='-5 5': ~ -510
    # n_bodies=5, joint_range='-3 3': ~ -633
    print('Total reward {}, min {}, max {}'.format(sum(reward_list), min(reward_list), max(reward_list)))
    env.close()


if __name__ == '__main__':
    test_random()
