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
    """ x: input_size = stimuli_size + q_size + q_vel_size
    external signal: first joint's target angle
    proprioceptive observations: joint angles and angular velocity
    """
    return [stimuli] + observation[1:12].tolist() + observation[15:26].tolist()


def y_func(action, **kwargs):
    """ y: action_size
    ctrl signal: joint action
    """
    return action.tolist()


if __name__ == '__main__':
    """
    data_size: the total amount of sequences is data_size * (max_episode_steps / seq_len)
    reset_noise_scale: noise ~ U[-scale, scale]
        sampled noise is added to initial q and q_vel (unit: radian)
        joint range [-100, 100] degrees -> [-1.745, 1.745] rad
    max_episode_steps: the same amount of time for adapting random init pose to sine pose
        how long it takes depends on reset_noise_scale
    """
    input_size = 23  # 1 + 11 + 11
    data_size = 9000
    seed = 42
    max_episode_steps = 128
    reset_noise_scale = 1.745
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale)
    model = Forward(dt=env.dt, seed=seed)
    dataset = generate_dataset(env, model, model_kwargs_func, x_func, y_func, input_size, data_size, seed, max_episode_steps)
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data.pt')
