""" dm_control swimmer with specific `n_links` to mimic C. elegans nematode """

import os
os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
os.environ['MUJOCO_GL'] = 'egl'  # glfw, egl, osmesa
os.environ['EGL_DEVICE_ID'] = '0'  # GPU id for rendering

import cv2
import numpy as np
from dm_control import suite
from swimmer_dm import grab_frame


def test_random():
    """ take random actions and record video """
    # env = suite.load(domain_name='swimmer', task_name='swimmer6', visualize_reward=True)  # task: swimmer6, swimmer15
    env = suite.swimmer.swimmer(n_links=3, random=None, environment_kwargs=None)  # time_limit=_DEFAULT_TIME_LIMIT

    frame = grab_frame(env)
    height, width, layers = frame.shape
    video = cv2.VideoWriter('video/swimmer_dm_v1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
    video.write(frame)

    action_spec = env.action_spec()
    time_step = env.reset()
    while not time_step.last():
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        time_step = env.step(action)
        # print(time_step.reward, time_step.discount, time_step.observation)
        video.write(grab_frame(env))

    # End render to video file
    video.release()


if __name__ == '__main__':
    test_random()
