""" OpenAI Gym Swimmer-v3 with specific `n_bodies` to mimic C. elegans nematode """

import gym
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from src.envs.mujoco.swimmer_gym_v3_v1 import swimmer


def test_random():
    """ take random actions and record video """
    # make env
    xml_str, xml_file = swimmer(n_bodies=5, xml_folder='src/envs/mujoco/assets/', xml_file='swimmer.xml')
    env = gym.make('Swimmer-v3', xml_file=os.path.join(os.getcwd(), xml_file))
    # remove xml file
    if os.path.exists(xml_file):
        os.remove(xml_file)
    # env specs
    print(env.action_space, env.action_space.low, env.action_space.high)
    print(env.observation_space, env.observation_space.low, env.observation_space.high)
    # record video
    os.makedirs('video', exist_ok=True)
    rec = VideoRecorder(env, base_path='video/swimmer_gym_v3_v1', enabled=True)  # Create the video recorder
    # run and record video
    observation = env.reset()
    rec.capture_frame()
    for i in range(1000):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rec.capture_frame()
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    rec.close()
    env.close()


if __name__ == '__main__':
    test_random()
