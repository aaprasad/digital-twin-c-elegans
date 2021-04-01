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


def grab_frame(env):
    # Get RGB rendering of env
    rgbArr = env.physics.render(height=240, width=320, camera_id=0, overlays=(), depth=False, segmentation=False, scene_option=None, render_flag_overrides=None)  # camera_id=-1
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)


def grab_frame_garage(env):
    rgb_arr = env.render(mode='rgb_array')
    return cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB)


def benchmarking_task_set():
    # Iterate over the whole task set: domain & task
    for domain_name, task_name in suite.BENCHMARKING:
        print(domain_name, task_name)
        env = suite.load(domain_name, task_name)


def test_random():
    env = suite.load(domain_name='swimmer', task_name='swimmer6', visualize_reward=True)  # task: swimmer6, swimmer15
    # env = suite.swimmer.swimmer(n_links=3, random=None, environment_kwargs=None)  # time_limit=_DEFAULT_TIME_LIMIT

    # Setup video writer - mp4 at 30 fps
    frame = grab_frame(env)
    height, width, layers = frame.shape
    video = cv2.VideoWriter('video/swimmer_dm.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
    video.write(frame)

    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    time_step = env.reset()
    while not time_step.last():
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        time_step = env.step(action)
        # print(time_step.reward, time_step.discount, time_step.observation)
        video.write(grab_frame(env))

    # End render to video file
    video.release()


def test_garage(framework: str, train: bool, log_dir: str):
    """
    dm_control swimmer metrics:
        avg reward over 100 consecutive episodes, maximum 1000 steps for each episode
    DeepMind Control Suite (https://arxiv.org/abs/1801.00690) benchmarks: A3C, D4PG, D4PG(Pixels), DDPG
        10^8 training steps:
            - swimmer6: 177.8 +- 7.8, 664.7 +- 11.1, 194.7 +- 15.9, 394.0 +- 14.1
            - swimmer15: 164.0 +- 7.3, 658.4 +- 10.0, 108.8 +- 11.9, 421.8 +- 13.5
        24 hours of training:
            - swimmer6: 526.4 +- 9.6, 651.0 +- 10.0, 168.9 +- 13.1, 461.0 +- 10.6
            - swimmer15: 196.8 +- 8.4, 681.1 +- 9.3, 146.0 +- 12.0, 410.9 +- 10.5
    100 episodes mean reward: swimmer6 with TRPO
        torch:
            - epochs 100, batch size 1024: 180.717
            - epochs 200, batch size 1024: 131.609
        tf:
            - epochs 40, batch size 4000: 119.254
            - epochs 200, batch size 1024: 127.148
    """
    from swimmer_gym_v3 import test_garage as test_garage_base
    from garage.envs.dm_control import DMControlEnv
    env = DMControlEnv.from_suite(domain_name='swimmer', task_name='swimmer6')
    test_garage_base(framework=framework, train=train, log_dir=log_dir, init_env=env)


def run_episode_cv2(env, policy, video_path):
    # Setup video writer
    frame = grab_frame_garage(env)
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
    video.write(frame)

    # run episodes
    observation, _ = env.reset()
    policy.reset()
    total_reward = 0.
    step = 0
    while True:
        # env.render()
        action, _ = policy.get_action(observation)
        env_step = env.step(action)
        video.write(grab_frame_garage(env))
        observation = env_step.observation
        total_reward += env_step.reward
        step += 1
        if env_step.last is True:
            break
    print("Episode finished after {} steps, reward: {}".format(step, total_reward))
    env.close()
    video.release()


def record_garage(framework: str):
    from swimmer_gym_v3 import record_garage as record_garage_base

    if framework == 'torch':
        record_garage_base(
            framework=framework, log_dir='log/swimmer_dm_trpo_torch', video_path='video/swimmer_dm_trpo_torch.mp4',
            run_episode=run_episode_cv2
        )
    elif framework == 'tf':
        record_garage_base(
            framework=framework, log_dir='log/swimmer_dm_trpo_tf', video_path='video/swimmer_dm_trpo_tf.mp4',
            run_episode=run_episode_cv2
        )
    else:
        raise AssertionError


if __name__ == '__main__':
    os.makedirs('video', exist_ok=True)

    # check out the task set
    # benchmarking_task_set()

    # take random action and record video
    # test_random()

    # run RL algos from garage with torch
    # test_garage(framework='torch', train=True, log_dir='log/swimmer_dm_trpo_torch')
    # test_garage(framework='torch', train=False, log_dir='log/swimmer_dm_trpo_torch')

    # run RL algos from garage with tf
    # test_garage(framework='tf', train=True, log_dir='log/swimmer_dm_trpo_tf')
    # test_garage(framework='tf', train=False, log_dir='log/swimmer_dm_trpo_tf')

    # load garage torch/tf checkpoint, and record video
    record_garage(framework='torch')
    # record_garage(framework='tf')
