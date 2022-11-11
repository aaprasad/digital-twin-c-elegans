"""
action space: Box(0.0, 1.0, (95,), float32)
observation space: Box(-inf, inf, (62,), float64)
    [0:56]: Ellipsoid2d-v0 observation space
    [56:59]: x-, y- and z-coordinates of the robot's center of mass (length, m)
    [59:62]: x-, y- and z-coordinates of the front tip (length, m)
"""

from analysis import get_results_numpy
import gym
import numpy as np
import os
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.models.muscle import ForwardPIDMuscle
from virtual_nematode.simulation import simulate


def action_func(model, step, observation, **kwargs):
    q = observation[4:28]
    # q_vel = observation[32:56]
    action = model.step(step, q=q)
    return action


def x_func(observation, **kwargs):
    # com = observation[56:58]
    return observation


if __name__ == '__main__':
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space)
    print(env.observation_space)
    # model = SinusoidalMuscle(dt=env.dt, n=25, a=0.02, freq=0.8, psi=0.05)
    # model = ForwardMuscle(dt=env.dt, n=25, a=30 * np.pi / 180, freq=0.8, psi=0.05, kp=1, kv=0)
    # model = ForwardPIDMuscle(
    #     dt=env.dt, n=25, a=40 * np.pi / 180, freq=0.8, psi=0.07,
    #     kp=1, kd=np.array([0.15 + i * 0.002 for i in range(24)])
    # )
    model = ForwardPIDMuscle(
        dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    x, y = simulate(env, model, action_func, x_func, seed=None, trials=100, render=False)
    os.makedirs('data', exist_ok=True)
    np.savez(os.path.join('data', 'simulate.npz'), x=x, y=y)
    get_results_numpy(x, y, max_episode_steps=max_episode_steps)
