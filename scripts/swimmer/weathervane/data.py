""" swimmer: chemotaxis based on weathervane mechanism """

import gymnasium as gym
import numpy as np
import os
import torch
from sim_vector import position_func, action_func
from virtual_nematode.envs.swimmer import make_chemotaxis_swimmers
from virtual_nematode.models.computational_model import ComputationalModelChemotaxis, ComputationalModelChemotaxisVector
from virtual_nematode.data.simulation import generate_dataset
from virtual_nematode.simulation import simulate_vector


def x_func(observation, vectorized=False, **kwargs):
    """ angle and angular velocity of the rotors, concentration, g, g_p, g_w """
    if vectorized is False:
        return observation[1:25].tolist() + observation[28:52].tolist() + observation[58:62].tolist()
    else:
        return np.hstack([observation[:, 1:25], observation[:, 28:52], observation[:, 58:62]])


def y_func(action, **kwargs):
    return action.tolist()


def data():
    """ generate data with torch multiprocessing """
    envs = make_chemotaxis_swimmers(
        seed=seed, trials=trials, distance=distance, position_func=position_func, n_bodies=25, joint_range='-100 100', body_len=0.1,
        max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale, camera_name=None, return_func=False
    )
    action_size = envs[0].action_space.shape[0]
    model = ComputationalModelChemotaxis(dt=envs[0].dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2., n_bias=25, **kwargs)
    envs = envs * data_size_per_trial
    data_size = len(envs)
    dataset = generate_dataset(envs, model, action_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed, disable=False)
    return dataset


def step_func(observation, action, **kwargs):
    """ data at each time step
    x: (batch_size, *), angle and angular velocity of the rotors, concentration, g, g_p, g_w
    y: (batch_size, action_size), action
    """
    x = torch.from_numpy(np.hstack([observation[:, 1:25], observation[:, 28:52], observation[:, 58:62]]))
    y = torch.from_numpy(action)
    return x, y


def done_func(result, **kwargs):
    """ data
    x: (batch_size, max_episode_steps, *)
    y: (batch_size, max_episode_steps, action_size)
    """
    x = torch.stack([rx for rx, _ in result], dim=1)
    y = torch.stack([ry for _, ry in result], dim=1)
    return x, y


def data_vector():
    """ generate data with gym.vector.AsyncVectorEnv multiprocessing """
    env = make_chemotaxis_swimmers(
        seed=seed, trials=trials, distance=distance, position_func=position_func, n_bodies=25, joint_range='-100 100', body_len=0.1,
        max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale, camera_name=None, return_func=True
    )
    env = env * data_size_per_trial
    env = gym.vector.AsyncVectorEnv(env)
    model = ComputationalModelChemotaxisVector(
        dt=env.get_attr('dt')[0], seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2., n_bias=25,
        batch_size=env.num_envs, **kwargs
    )
    x, y = simulate_vector(env, model, action_func, step_func, done_func, seed=11, render=False)  # (batch_size, max_episode_steps, 1)
    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset


if __name__ == '__main__':
    input_size = 52
    trials = 100  # number of envs with different source positions
    data_size_per_trial = 30  # number of trials for each env
    seed = 11
    max_episode_steps = 3500
    distance = 15
    reset_noise_scale = 1.745
    kwargs = {'backward': False, 'omega': False, 'weathervane': True, 'random_walk': False, 'weathervane_reverse': False}
    dataset = data()  # dataset = data_vector()
    print('dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data.pt')
