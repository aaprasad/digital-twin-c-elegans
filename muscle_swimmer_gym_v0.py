""" muscle swimmer implemented based on OpenAI Gym Swimmer-v3 """

from src.envs.mujoco.swimmer_gym_v3_v0 import SwimmerEnv
from src.envs.mujoco.muscle_swimmer_gym_v0 import swimmer
import gym
import mujoco_py
import numpy as np
import os


def test_random():
    """ take random actions """
    xml_str = swimmer(xml_file='src/envs/mujoco/assets/swimmer.xml')
    # build model
    model = mujoco_py.load_model_from_xml(xml_str.decode("utf-8"))
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    sim_state = sim.get_state()
    while True:
        sim.set_state(sim_state)
        for i in range(1000):
            for j in range(len(sim.data.ctrl)):
                sim.data.ctrl[j] = np.random.random()
            sim.step()
            viewer.render()
        if os.getenv('TESTING') is not None:
            break


def make_swimmer():
    """ create swimmer env """
    xml_str = swimmer(xml_file='src/envs/mujoco/assets/swimmer.xml')
    env = SwimmerEnv(xml_str=xml_str.decode('utf-8'))
    # action: Box(0.0, 1.0, (4,), float32), muscles' actions
    print(env.action_space, env.action_space.low, env.action_space.high)
    # observation: Box(-inf, inf, (8,), float64), the same as Swimmer-v3
    # the states of tendon/muscle need to be added
    print(env.observation_space, env.observation_space.low, env.observation_space.high)
    return env


def test_random_gym():
    """ take random actions with gym env """
    # make env
    env = make_swimmer()
    # record video
    env = gym.wrappers.Monitor(env, directory='video/muscle_swimmer_gym_v0', force=True)
    # run and record video
    observation = env.reset()
    for i in range(1000):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} steps'.format(i + 1))
            break
    env.close()


if __name__ == '__main__':
    # test_random()
    test_random_gym()
