"""
Swimmer with n_links from dm_control
"""

import os
os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
os.environ['MUJOCO_GL'] = 'egl'  # glfw, egl, osmesa
os.environ['EGL_DEVICE_ID'] = '0'  # GPU id for rendering

import cv2
import numpy as np
from dm_control import suite


def grabFrame(env):
    # Get RGB rendering of env
    rgbArr = env.physics.render(height=240, width=320, camera_id=0, overlays=(), depth=False, segmentation=False, scene_option=None, render_flag_overrides=None)  # camera_id=-1
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)


def benchmarking_task_set():
    # Iterate over the whole task set: domain & task
    for domain_name, task_name in suite.BENCHMARKING:
        print(domain_name, task_name)
        env = suite.load(domain_name, task_name)


def test_random():
    # env = suite.load(domain_name='swimmer', task_name='swimmer6', visualize_reward=True)  # task: swimmer6, swimmer15
    env = suite.swimmer.swimmer(n_links=3, random=None, environment_kwargs=None)  # time_limit=_DEFAULT_TIME_LIMIT

    # Setup video writer - mp4 at 30 fps
    frame = grabFrame(env)
    height, width, layers = frame.shape
    os.makedirs('video', exist_ok=True)
    video = cv2.VideoWriter('video/swimmer_dm.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
    video.write(frame)

    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    time_step = env.reset()
    while not time_step.last():
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        time_step = env.step(action)
        # print(time_step.reward, time_step.discount, time_step.observation)
        video.write(grabFrame(env))

    # End render to video file
    video.release()


def train_garage():
    from garage.envs.dm_control import DMControlEnv

    """ make env """
    env = DMControlEnv.from_suite(domain_name='swimmer', task_name='swimmer6')

    return env, model


def test_garage():
    env, model = train_garage()


if __name__ == '__main__':
    # check out the task set
    # benchmarking_task_set()

    # take random action and record video
    test_random()

    # run RL algos from garage
    # test_garage()
