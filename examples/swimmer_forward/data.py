""" swimmer: open-loop control of forward locomotion
generate forward simulation dataset
"""

import os
from sim import make_swimmer, model_kwargs_func
import torch
from virtual_nematode.data.simulation import generate_dataset
from virtual_nematode.models.forward import Forward


def x_func(stimuli, **kwargs):
    return [stimuli]


if __name__ == '__main__':
    """
    x: torch.Tensor, (max_episode_steps, input_size)
        action sequence of the first joint in trials
    y: torch.Tensor, (max_episode_steps, action_size)
        action sequences in trials
    """
    input_size = 1
    data_size = 1
    reset_noise_scale = 0.
    max_episode_steps = 5000
    seed = 42
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale)
    model = Forward(dt=env.dt, seed=seed)
    dataset = generate_dataset(env, model, model_kwargs_func, x_func, input_size, data_size, seed, max_episode_steps)
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data.pt')
