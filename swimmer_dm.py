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
    env = suite.load(domain_name='swimmer', task_name='swimmer6', visualize_reward=True)  # task: swimmer6, swimmer15
    # env = suite.swimmer.swimmer(n_links=3, random=None, environment_kwargs=None)  # time_limit=_DEFAULT_TIME_LIMIT

    # Setup video writer - mp4 at 30 fps
    frame = grabFrame(env)
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
        video.write(grabFrame(env))

    # End render to video file
    video.release()


def test_garage(framework: str, train: bool, log_dir: str):
    from swimmer_gym_v3 import test_garage as test_garage_base
    from garage.envs.dm_control import DMControlEnv
    env = DMControlEnv.from_suite(domain_name='swimmer', task_name='swimmer6')
    test_garage_base(framework=framework, train=train, log_dir=log_dir, init_env=env)


def run_episode(env, policy, video_path):
    # Setup video writer
    frame = grabFrame(env)
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
    video.write(frame)

    # run episodes
    time_step = env.reset()
    policy.reset()
    total_reward = 0.
    step = 0
    while not time_step.last():
        # env.render()
        action, _ = policy.get_action(time_step.observation)
        time_step = env.step(action)
        video.write(grabFrame(env))
        total_reward += time_step.reward
        step += 1
    print("Episode finished after {} steps, Reward: {}".format(step, total_reward))
    env.close()
    video.release()


def record_garage(framework: str):
    from garage.experiment import Snapshotter

    def record_garage_torch():
        # Load the env and policy from snap-shot
        snapshotter = Snapshotter()
        data = snapshotter.load(log_dir='log/swimmer_dm_trpo_torch', itr='last')  # itr: iteration to load, an integer, 'last' or 'first'
        env = data['env']
        policy = data['algo'].policy
        run_episode(env=env, policy=policy, video_path='video/swimmer_dm_trpo_torch.mp4')

    def record_garage_tf():
        import tensorflow as tf

        # run episodes
        snapshotter = Snapshotter()
        with tf.compat.v1.Session():
            data = snapshotter.load(log_dir='log/swimmer_dm_trpo_tf', itr='last')
            env = data['env']
            policy = data['algo'].policy
            run_episode(env=env, policy=policy, video_path='video/swimmer_dm_trpo_tf.mp4')

    if framework == 'torch':
        record_garage_torch()
    elif framework == 'tf':
        record_garage_tf()
    else:
        raise AssertionError


if __name__ == '__main__':
    os.makedirs('video', exist_ok=True)

    # check out the task set
    # benchmarking_task_set()

    # take random action and record video
    # test_random()

    # run RL algos from garage with torch, 100 episodes mean reward: 180.71739610587625
    # test_garage(framework='torch', train=True, log_dir='log/swimmer_dm_trpo_torch')
    # test_garage(framework='torch', train=False, log_dir='log/swimmer_dm_trpo_torch')

    # run RL algos from garage with tf, 100 episodes mean reward: 119.2544067832367
    # test_garage(framework='tf', train=True, log_dir='log/swimmer_dm_trpo_tf')
    # test_garage(framework='tf', train=False, log_dir='log/swimmer_dm_trpo_tf')

    # load garage torch/tf checkpoint, and record video
    record_garage(framework='torch')
    # record_garage(framework='tf')
