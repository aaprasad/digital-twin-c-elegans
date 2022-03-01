""" swimmer: chemotaxis based on weathervane mechanism """

import os
import torch
from sim import position_func, action_func
from virtual_nematode.envs.swimmer import make_chemotaxis_swimmer
from virtual_nematode.models.computational_model import ComputationalModelChemotaxis
from virtual_nematode.data.simulation import generate_dataset
from virtual_nematode.data.concat import ConcatDataset


def x_func(observation, **kwargs):
    """ angle and angular velocity of the rotors, concentration, g, g_p, g_w """
    return observation[1:25].tolist() + observation[28:52].tolist() + observation[58:62].tolist()


def y_func(action, **kwargs):
    return action.tolist()


if __name__ == '__main__':
    input_size = 52
    data_size_per_trial = 10
    trials = 100
    seed = 11
    max_episode_steps = 3500
    reset_noise_scale = 1.745
    envs = make_chemotaxis_swimmer(
        seed=seed, trials=trials, distance=15, position_func=position_func, n_bodies=25, joint_range='-100 100', body_len=0.1,
        max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale, camera_name=None
    )
    action_size = envs[0].action_space.shape[0]
    kwargs = {'backward': False, 'omega': False, 'weathervane': True, 'random_walk': False, 'weathervane_reverse': False}
    model = ComputationalModelChemotaxis(dt=envs[0].dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2., **kwargs)
    dataset = []
    for i, env in enumerate(envs):
        print(i, env.source.tolist(), end=' ')
        d = generate_dataset(env, model, action_func, x_func, y_func, input_size, action_size, data_size_per_trial, max_episode_steps, seed, disable=True)
        dataset.append(d)
    dataset = ConcatDataset(dataset)
    print('dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data.pt')
