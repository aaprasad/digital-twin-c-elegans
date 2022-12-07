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


if __name__ == '__main__':
    max_episode_steps = 2500
    seed = 7  # None
    env = make_swimmer_weathervane(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        distance=15, source=(0, 0), position_func=position_func,  # distance = 3 * sigma
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    """
    env = make_swimmer_weathervane_fixed(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        pos=(15, 0), source=(0, 0), position_func=position_func,  # distance = 3 * sigma
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    """
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space, env.observation_space)
    print(env.source)
    # model = WeathervanePIDMuscle(
    #     k_w=1, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
    #     kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
    #     kd=0.15
    # )
    model = WeathervanePIDMuscleDelay(
        k_w=1, delay_step=100, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    x, y = simulate(env, model, action_func, x_func, seed=seed, trials=100, render=False)
    os.makedirs('data', exist_ok=True)
    np.savez(os.path.join('data', 'simulate.npz'), x=x, y=y)
    # np.savez(os.path.join('data', 'simulate_3sigma.npz'), x=x, y=y)
    get_results_numpy(x, y, max_episode_steps=max_episode_steps, sigma=5)
