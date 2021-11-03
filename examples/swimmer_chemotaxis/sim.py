""" chemotaxis environment """

import gym
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.swimmer_v3_v1 import swimmer
from gym_worm.envs.mujoco.chemotaxis import chemotaxis
from gym_worm.wrappers.distribution_observation import DistributionObservation
from gym_worm.wrappers.recorder import Recorder
import numpy as np
import os
from virtual_nematode.models.computational_model import ComputationalModelChemotaxis


def fick(target, source, sigma=5):
    """ Fick's second law: calculate target position's value in a source distribution
    C = N_0 * exp^{-r ^ 2 / 400Dt}/4\pi dDt (gaussian kernel)
    t: ignore the fact that concentration changes through time because of diffusion
    sigma: defines the range where gradient exists so that gradient-based navigation works
    Reference:
        https://doi.org/10.1523/JNEUROSCI.3633-08.2009
        https://doi.org/10.1038/s41598-018-35157-1
    """
    r = np.linalg.norm(target - source)  # Euclidean distance
    c = np.exp(-r ** 2 / (2 * sigma ** 2))  # gaussian kernel
    return c


def make_swimmer(
    n_bodies=12, joint_range='-100 100', body_len=0.25, camera_pos='0 -6 6', camera_z=None, camera_name=None,
    max_episode_steps=1000, x=0, y=0, video_name='swimmer'
):
    """ create swimmer env: multibody model
    radius=0.04mm, body_len=0.1mm, n_bodies=12, q_max=0.69rad (~39.53409 degrees)
    joint_size=0.1 (radius) -> body_len=0.25
    whole body length = 0.25 * 12 = 3
    References
        https://doi.org/10.1038/s41598-018-35157-1
    """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str = camera(xml_str, camera_pos, camera_z)
    xml_str = chemotaxis(xml_str, x, y)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'))
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = DistributionObservation(env, dt=env.dt, f=fick, source=np.array([x, y]))
    env = Recorder(env, stats_name=['concentration'], camera_name=camera_name)
    if camera_name is not None:
        env = gym.wrappers.Monitor(env, directory=os.path.join('video', video_name), force=True)
    return env


def simulate(seed=None):
    """ control by sinusoidal motion
    seed: env simulation stays the same with seeding
    """
    env = make_swimmer(max_episode_steps=2500, x=9, y=12, camera_z=50, camera_name=None)  # distance from source: 15
    env.seed(seed)
    observation = env.reset()
    info = {'g_p': 0., 'g_w': 0.}
    model = ComputationalModelChemotaxis(dt=env.dt, seed=seed)
    for i in range(10 ** 6):
        # env.render()
        action = model.step(step=i, q=observation[1:12], q_vel=observation[15:26], g_p=info['g_p'], g_w=info['g_w'])
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} steps'.format(i + 1))
            break
    print('chemotaxis index', np.mean(env.stats['concentration']))
    env.close()


if __name__ == '__main__':
    simulate()
