""" swimmer: chemotaxis
generate concatenated chemotaxis dataset with several chemical source positions
"""

import os
from virtual_nematode.data.simulation import generate_dataset
from virtual_nematode.models.computational_model import ComputationalModelChemotaxis
from virtual_nematode.utils import clock_position
from sim import make_swimmer
import torch


def model_kwargs_func(observation, **kwargs):
    return {'q': observation[1:12], 'q_vel': observation[15:26], 'g_p': observation[28], 'g_w': observation[29]}


def x_func(observation, **kwargs):
    """ x: input_size = 1
    external signal: concentrations sensed at nose tip
    """
    return [observation[26]]


def y_func(action, **kwargs):
    """ y: action_size
    ctrl signal: joint action
    """
    return action.tolist()


if __name__ == '__main__':
    """
    data_size: the total dataset size, should be divided for each env (with different source position)
    seed: the randomly generated dataset stays the same with seeding
    """
    input_size = 1
    data_size = 12000
    max_episode_steps = 2500
    distance = 15
    seed = 42
    envs = [make_swimmer(max_episode_steps=max_episode_steps, x=x, y=y) for x, y in clock_position(distance=distance)]
    models = [ComputationalModelChemotaxis(dt=env.dt) for env in envs]
    data_size = data_size // len(envs)
    datasets = []
    for env, model in zip(envs, models):
        action_size = env.action_space.shape[0]
        dataset = generate_dataset(env, model, model_kwargs_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed)
        print(env.source.tolist(), len(dataset), dataset[0][0].size(), dataset[0][1].size())
        datasets.append(dataset)
    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    print('dataset', len(concat_dataset), concat_dataset[0][0].size(), concat_dataset[0][1].size())
    os.makedirs('data', exist_ok=True)
    torch.save(concat_dataset, 'data/data.pt')
