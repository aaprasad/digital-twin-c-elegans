""" swimmer: forward locomotion """

import gym
import numpy as np
from virtual_nematode.envs.swimmer import make_swimmer_friction
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation import simulate


def action_func(model, step, observation, **kwargs):
    q = observation[2:26]
    q_vel = observation[30:54]
    action = model.step(step, q, q_vel)
    return action


def step_func(observation, **kwargs):
    com = observation[52:54]
    return com


def done_func(index, result, **kwargs):
    displacement = np.linalg.norm(np.array(result[-1]) - np.array(result[0]), ord=2)
    print('Trial {}: com displacement {:.2f}'.format(index + 1, displacement))
    return displacement


if __name__ == '__main__':
    max_episode_steps = 2500
    env = make_swimmer_friction(
        n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=max_episode_steps, reset_noise_scale=0.1
    )
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False)
