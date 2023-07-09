"""
action space: Box(0.0, 1.0, (95,), float32)
observation space: Box(-inf, inf, (62,), float64)
    [0:56]: Ellipsoid2d-v0 observation space
    [56:59]: x-, y- and z-coordinates of the robot's center of mass (length, m)
    [59:62]: x-, y- and z-coordinates of the front tip (length, m)
    [62:66]: distribution related observation
"""

from analysis import get_results_numpy
import gym
import math
import numpy as np
import os
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer_weathervane, make_swimmer_weathervane_fixed
from virtual_nematode.models.muscle import WeathervanePIDMuscle, WeathervanePIDMuscleDelay
from virtual_nematode.simulation import simulate


def position_func(observation, **kwargs):
    """ 2D center of mass and position of the first body segment """
    com, position = observation[56:58], observation[59:61]
    return com, position


def action_func(model, step, observation, **kwargs):
    q = observation[4:28]
    # q_vel = observation[32:56]
    # g_p = observation[64]
    g_w = observation[65]
    action = model.step(step, q=q, g_w=g_w)
    return action


def x_func(observation, **kwargs):
    # com, position = observation[56:58], observation[59:61]
    # gradient = observation[62:66]  # c, g, g_p, g_w
    return observation


def simulate_random(k_w=1):
    env = make_swimmer_weathervane(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        distance=15, source=(0, 0), position_func=position_func,  # distance = 3 * sigma
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space, env.observation_space)
    print(env.source)
    model = WeathervanePIDMuscle(
        k_w=k_w, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    # model = WeathervanePIDMuscleDelay(
    #     k_w=1, delay_step=100, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
    #     kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
    #     kd=0.15
    # )
    x, y = simulate(env, model, action_func, x_func, seed=seed, trials=trials, render=False)
    np.savez(os.path.join('data', 'simulate_random_kw{}.npz'.format(k_w)), x=x, y=y)
    get_results_numpy(x, y, max_episode_steps=max_episode_steps, sigma=5)


def simulate_fixed(pos=(15, 0), k_w=1):
    env = make_swimmer_weathervane_fixed(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        pos=pos, source=(0, 0), position_func=position_func,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space, env.observation_space)
    print(env.source)
    model = WeathervanePIDMuscle(
        k_w=k_w, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    x, y = simulate(env, model, action_func, x_func, seed=seed, trials=trials, render=False)
    distance = int(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
    np.savez('data/simulate_d{}_kw{}.npz'.format(distance, k_w), x=x, y=y)
    get_results_numpy(x, y, max_episode_steps=max_episode_steps, sigma=5)


def simulate_once_fixed(pos=(15, 0), k_w=1, angle=None, camera_name=None, camera_z=50):
    env = make_swimmer_weathervane_fixed(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        pos=pos, source=(0, 0), position_func=position_func,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic',
        angle=angle, camera_pos='-1.25 0 2.5', camera_name=camera_name, camera_z=camera_z
    )
    d = int(math.sqrt(pos[0] ** 2 + pos[1] ** 1))
    save_path = 'data/simulate_d{}_a{:.2f}_{}'.format(d, angle, camera_name)
    env = gym.wrappers.Monitor(env, directory=save_path, force=True)
    print(env.action_space, env.observation_space)
    print(env.source)
    model = WeathervanePIDMuscle(
        k_w=k_w, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    x, y = simulate(env, model, action_func, x_func, seed=seed, trials=1, render=False)
    np.savez(os.path.join(save_path, 'data.npz'), x=x, y=y)
    get_results_numpy(x, y, max_episode_steps=max_episode_steps, sigma=5)


if __name__ == '__main__':
    trials = 100
    max_episode_steps = 2500
    seed = 7  # None
    os.makedirs('data', exist_ok=True)
    """ weathervane controller """
    simulate_random(k_w=1)
    # simulate_fixed(pos=(15, 0), k_w=1)
    # simulate_fixed(pos=(10, 0), k_w=1)
    # simulate_fixed(pos=(5, 0), k_w=1)
    """ forward controller """
    # simulate_random(k_w=0)
    # simulate_fixed(pos=(15, 0), k_w=0)
    # simulate_fixed(pos=(10, 0), k_w=0)
    # simulate_fixed(pos=(5, 0), k_w=0)
    """ recording """
    # simulate_once_fixed(pos=(-10, 0), k_w=1, angle=np.pi/2.)
    # simulate_once_fixed(pos=(-10, 0), k_w=1, angle=np.pi / 2., camera_name='fixedcam', camera_z=32)
