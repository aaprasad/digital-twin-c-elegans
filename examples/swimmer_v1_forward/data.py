""" swimmer: closed-loop (feedback) control of forward locomotion
generate forward simulation dataset
"""

import os
import torch
from sim import model_kwargs_func
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.data.simulation import generate_dataset


def x_func(stimuli, observation, **kwargs):
    return [stimuli] + observation[1:25].tolist() + observation[28:].tolist()


def y_func(action, **kwargs):
    return action.tolist()


if __name__ == '__main__':
    """
    max_episode_steps: the same amount of time for adapting random init pose to sine pose
        how long it takes depends on reset_noise_scale
    x: torch.Tensor, (max_episode_steps, input_size)
    y: torch.Tensor, (max_episode_steps, action_size)
    """
    input_size = 49  # 1 + 24 + 24
    data_size = 4500
    seed = 42
    max_episode_steps = 256
    reset_noise_scale = 1.745
    env = make_swimmer(
        n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=max_episode_steps,
        reset_noise_scale=reset_noise_scale
    )
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    dataset = generate_dataset(env, model, model_kwargs_func, x_func, y_func, input_size, data_size, seed, max_episode_steps)
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data.pt')
