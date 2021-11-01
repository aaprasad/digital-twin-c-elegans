""" swimmer: closed-loop (feedback) control of forward locomotion
generate forward simulation dataset
"""

import os
from virtual_nematode.data.simulation import SimulationDataset
from virtual_nematode.models.forward import Forward
from virtual_nematode.utils import sample_seed
from virtual_nematode.envs.swimmer_v3_v1 import make_swimmer
import torch


def generate_sample(env, model, mode):
    """ run a forward movement simulation
    x: torch.Tensor, (max_episode_steps, stimuli_size + q_size + q_vel_size)
    y: torch.Tensor, (max_episode_steps, action_size)
    mode: stimuli mode
    """
    seed = sample_seed()
    env.seed(seed)  # seed env
    model.seed(seed)  # seed model
    observation = env.reset()
    x = []
    y = []
    for i in range(10 ** 6):
        action = model.step(step=i, q=observation[1:12], q_vel=observation[15:])
        stimuli = model.stimuli(step=i, mode=mode)
        x.append([stimuli] + observation[1:12].tolist() + observation[15:].tolist())
        y.append(action.tolist())
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    return x, y


def generate_dataset(
    data_size=9000, seed=42, max_episode_steps=128, reset_noise_scale=1.745, mode='sine_wave', save_name='target.pt'
):
    """ generate forward movement dataset
    x: external signal (first joint's target angle), proprioceptive observations (joint angles and angular velocity)
    y: actions
    data_size: the total amount of sequences is data_size * (max_episode_steps / seq_len)
    max_episode_steps: the same amount of time for adapting random init pose to sine pose
        how long it takes depends on reset_noise_scale
    reset_noise_scale: noise ~ U[-scale, scale]
        sampled noise is added to initial q and q_vel (unit: radian)
        joint range [-100, 100] degrees -> [-1.745, 1.745] rad
    """
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale)
    model = Forward(dt=env.dt, seed=seed)
    input_size = 23  # 1 + 11 + 11
    action_size = env.action_space.shape[0]
    dataset = SimulationDataset(
        data_size, max_episode_steps, input_size, action_size, seed, generate_sample,
        # simulation fn kwargs
        env=env, model=model, mode=mode
    )
    print('dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    """ results
    dataset: x [9000, 128, 23], y [9000, 128, 11]
    """
    generate_dataset(mode='sine_wave', save_name='data.pt')
    # generate_dataset(mode='square_wave', save_name='data.pt')
