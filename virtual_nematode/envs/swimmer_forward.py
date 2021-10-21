import gym
from gym_worm.envs.mujoco.swimmer_v3_v2 import swimmer
from gym_worm.wrappers.position import Position
from gym_worm.wrappers.recorder import Recorder
import numpy as np
import os
from virtual_nematode.models.forward import Forward


def make_swimmer(
    n_bodies=12, joint_range='-100 100', body_len=0.25, camera_pos='0 -6 6', camera_z=None, camera_name=None,
    max_episode_steps=1000, video_name='swimmer_forward', reset_noise_scale=0.1
):
    """ create swimmer env """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len, camera_pos, camera_z)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = Position(env)
    env = Recorder(env, stats_name=['com'], camera_name=camera_name)
    if camera_name is not None:
        env = gym.wrappers.Monitor(env, directory=os.path.join('video', video_name), force=True)
    return env


def simulate(seed=None, max_episode_steps=2500, trials=1):
    """ forward sinusoidal movement """
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=0.1, camera_z=50, camera_name=None)
    displacements = []
    if trials > 1:
        seed = None  # ensure that seed is different in each trial
    for i in range(trials):
        env.seed(seed)
        observation = env.reset()
        model = Forward(dt=env.dt, seed=seed)
        for step in range(10 ** 6):
            # env.render()
            action = model.step(step=step, q=observation[1:12], q_vel=observation[15:])
            observation, reward, done, info = env.step(action)
            if done:
                d = np.linalg.norm(np.array(env.stats['com'][-1]) - np.array(env.stats['com'][0]), ord=2)
                displacements.append(d)
                print('Trial {}: com displacement {:.2f} / {} steps'.format(i + 1, d, step + 1))
                break
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(displacements), np.mean(displacements), max_episode_steps))
