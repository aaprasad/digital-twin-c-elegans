""" swimmer: chemotaxis """

import gym
import numpy as np
import os
from src.envs.mujoco.swimmer_gym_v3_v0 import SwimmerEnv
from src.envs.mujoco.swimmer_gym_v3_v2 import swimmer
from src.envs.mujoco.chemotaxis import chemotaxis
from src.wrappers.distribution import Distribution
from src.wrappers.recorder import Recorder
from src.models.computational_model import ComputationalModelChemotaxis


def fick(target, source, sigma=5):
    """ calculate target position's value in a source distribution
    - Fick's laws of diffusion: Fick's second law
    - Its fundamental solution: C = N_0 * exp^{-r ^ 2 / 400Dt}/4\pi dDt
    - Its fundamental solution is the same as gaussian kernel
    - Ignore the fact that concentration changes through time because of diffusion: concentration doesn't change through time
    - sigma: defines the range where gradient exists so that gradient-based navigation works
    Reference:
        - Parallel Use of Two Behavioral Mechanisms for Chemotaxis in Caenorhabditis elegans
        - A computational model of internal representations of chemical gradients in environments for chemotaxis of Caenorhabditis elegans
    """
    r = np.linalg.norm(target - source)  # Euclidean distance
    c = np.exp(-r ** 2 / (2 * sigma ** 2))  # gaussian kernel
    return c


def make_swimmer(
    n_bodies=12, joint_range='-100 100', body_len=0.25, camera_pos='0 -6 6', camera_z=None, camera_name=None,
    max_episode_steps=1000, x=0, y=0, video_name='swimmer_chemotaxis'
):
    """ create swimmer env: multibody model
    - radius=0.04mm, body_len=0.1mm, n_bodies=12, q_max=0.69rad (~39.53409 degrees)
    - joint_size=0.1 (radius) -> body_len=0.25
    - whole body length = 0.25 * 12 = 3
    - citation: A computational model of internal representations of chemical gradients in environments for chemotaxis of Caenorhabditis elegans
    """
    # generate xml str
    xml_str = swimmer(n_bodies=n_bodies, joint_range=joint_range, body_len=body_len, xml_file='src/envs/mujoco/assets/swimmer.xml', camera_pos=camera_pos, camera_z=camera_z)
    xml_str = chemotaxis(xml_str=xml_str, x=x, y=y)
    # make and wrap env
    env = SwimmerEnv(xml_str=xml_str.decode('utf-8'))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = Distribution(env, dt=env.dt, f=fick, source=np.array([x, y]))
    env = Recorder(env, stats_name=['concentration'], camera_name=camera_name)
    # record video: if force is True, clear previous monitor files
    if camera_name is not None:
        env = gym.wrappers.Monitor(env, directory=os.path.join('video', video_name), force=True)
    return env


def test_random():
    """ take random actions """
    d = 15  # distance from source
    x = np.random.uniform(-d, d)
    y = np.sqrt(d ** 2 - x ** 2)
    env = make_swimmer(x=x, y=y)
    observation = env.reset()
    for i in range(10 ** 6):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    env.close()


def test_sinusoidal_motion(seed=None):
    """ control by sinusoidal motion
    seed: env simulation stays the same with seeding
    """
    env = make_swimmer(max_episode_steps=2500, x=9, y=12, camera_z=50, camera_name=None)  # distance from source: 15
    env.seed(seed)
    observation = env.reset()
    info = {'g_p': 0., 'g_w': 0.}
    model = ComputationalModelChemotaxis(dt=env.dt, seed=seed)
    for i in range(10 ** 6):
        env.render()
        action = model.step(step=i, q=observation[1:12], q_vel=observation[15:], g_p=info['g_p'], g_w=info['g_w'])
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    print('chemotaxis index', np.mean(env.stats['concentration']))
    env.close()


if __name__ == '__main__':
    # test_random()
    test_sinusoidal_motion()
