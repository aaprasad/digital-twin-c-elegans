""" OpenAI Gym Swimmer-v3 with specific `n_bodies` to mimic C. elegans nematode """

import gym
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from src.envs.mujoco.swimmer_gym_v3_v1 import swimmer


def make_swimmer(n_bodies, camera_pos):
    """ create swimmer env """
    # make env
    xml_str, xml_file = swimmer(n_bodies=n_bodies, xml_folder='src/envs/mujoco/assets/', xml_file='swimmer.xml', camera_pos=camera_pos)
    env = gym.make('Swimmer-v3', xml_file=os.path.join(os.getcwd(), xml_file))
    # remove xml file
    if os.path.exists(xml_file):
        os.remove(xml_file)
    # action: Box(-1.0, 1.0, (4,), float32), torque control of the joints, [5 bodies]
    print(env.action_space, env.action_space.low, env.action_space.high)
    # observation: Box(-inf, inf, (12,), float64), qpos[2:7] + qvel[0:7], [5 bodies]
    # qpos[0:7]: x pos + y pos + ? + joint1~4 angle, [5 bodies]
    # qvel[0:7]: x vel + y vel + ? + joint1~4 vel, [5 bodies]
    print(env.observation_space, env.observation_space.low, env.observation_space.high)
    return env


def test_random():
    """ take random actions and record video """
    # make env
    env = make_swimmer(n_bodies=5, camera_pos='0 -6 6')
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
