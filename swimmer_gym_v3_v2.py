""" OpenAI Gym Swimmer-v3 with specific `n_bodies` and `body_len` to mimic C. elegans nematode """

import gym
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from src.envs.mujoco.swimmer_gym_v3_v2 import swimmer


def make_swimmer(n_bodies, body_len, camera_pos):
    """ create swimmer env """
    xml_folder = 'src/envs/mujoco/assets/'
    xml_str = swimmer(n_bodies=n_bodies, body_len=body_len, xml_file=os.path.join(xml_folder, 'swimmer.xml'), camera_pos=camera_pos)
    xml_file = os.path.join(xml_folder, 'swimmer_temp.xml')
    with open(xml_file, 'wb') as f:
        f.write(xml_str)
    env = gym.make('Swimmer-v3', xml_file=os.path.join(os.getcwd(), xml_file))
    if os.path.exists(xml_file):
        os.remove(xml_file)
    print(env.action_space, env.action_space.low, env.action_space.high)
    print(env.observation_space, env.observation_space.low, env.observation_space.high)
    return env


def test_random():
    """ take random actions and record video """
    # make env
    env = make_swimmer(n_bodies=5, body_len=0.5, camera_pos='0 -6 6')
    # record video
    os.makedirs('video', exist_ok=True)
    rec = VideoRecorder(env, base_path='video/swimmer_gym_v3_v2', enabled=True)  # Create the video recorder
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
