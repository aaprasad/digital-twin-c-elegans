"""
Swimmer-Gym-v3-v0 is the same as Swimmer-v3 from OpenAI Gym
- xml_file is configured to load local .xml file.
- swimmer.xml is unedited from Gym.
"""

import gym
import time
from gym.envs.registration import register
from gym.wrappers.monitoring.video_recorder import VideoRecorder


if __name__ == '__main__':
    """ register and make env """
    register(
        id='Swimmer-Gym-v3-v0',
        entry_point='src.envs.mujoco.swimmer_gym_v3_v0:SwimmerEnv',
        max_episode_steps=1000,  # an episode will end after 1000 steps or when the agent reaches end states.
        reward_threshold=360.0,  # if avg reward over 100 consecutive episodes >= 360.0, then it's solved!
    )
    env = gym.make('Swimmer-Gym-v3-v0')

    """ record video
    - there's a relevant bug in gym==0.18.0, use 'pip install -e .' to install dev version instead
    - video will be named: based_path + '.mp4'
    """
    rec = VideoRecorder(env, base_path='swimmer_gym_v3_v0', enabled=True)  # Create the video recorder

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
