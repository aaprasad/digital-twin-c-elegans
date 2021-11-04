""" swimmer: closed-loop (feedback) control of forward locomotion
swimmer configuration: n_bodies = 25
"""

import os
import torch
from sim import model_kwargs_func
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.data.simulation import generate_dataset


def x_func(stimuli, observation, **kwargs):
    """ x: input_size = stimuli_size + q_size + q_vel_size """
    return [stimuli] + observation[1:25].tolist() + observation[28:52].tolist()


def y_func(action, **kwargs):
    """ y: action_size = 24 """
    return action.tolist()


if __name__ == '__main__':
    input_size = 49  # 1 + 24 + 24
    data_size = 6000
    seed = 42
    max_episode_steps = 192
    reset_noise_scale = 1.745
    env = make_swimmer(
        n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=max_episode_steps,
        reset_noise_scale=reset_noise_scale
    )
    action_size = env.action_space.shape[0]
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    dataset = generate_dataset(env, model, model_kwargs_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed)
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data.pt')
