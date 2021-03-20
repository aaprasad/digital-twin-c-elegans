"""
AntBulletEnv-v0 from PyBullet
"""

import gym
# import pybullet
import pybullet_envs
from gym.wrappers.monitoring.video_recorder import VideoRecorder


if __name__ == '__main__':
    """ register and make env """
    env = gym.make('AntBulletEnv-v0')

    """ record video
    - there's a relevant bug in gym==0.18.0, use 'pip install -e .' to install dev version instead
    - video will be named: based_path + '.mp4'
    """
    rec = VideoRecorder(env, base_path='antbulletenv_v0', enabled=True)  # Create the video recorder

    """ run and record """
    observation = env.reset()
    # print(observation)
    rec.capture_frame()  # Capture the starting position
    for i in range(1000):
        # env.render()  # show the current frame of visualization
        action = env.action_space.sample()  # sample a random action
        observation, reward, done, info = env.step(action)
        # print(observation)
        rec.capture_frame()
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    rec.close()  # Close the recording
    env.close()
