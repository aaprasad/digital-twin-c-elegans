""" swimmer: open-loop control of forward locomotion
generate forward simulation dataset
"""

import os
from virtual_nematode.data.simulation import SimulationDataset
from virtual_nematode.models.forward import Forward
from virtual_nematode.utils import sample_seed
from swimmer_forward import make_swimmer
import torch


def generate_sample(env, model, mode):
    """ run a forward movement simulation
    x: torch.Tensor, (max_episode_steps, 1)
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
        x.append(stimuli)
        y.append(action.tolist())
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    x = torch.tensor(x, dtype=torch.float64).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float64)
    return x, y


def generate_dataset(data_size=100, seed=42, max_episode_steps=2500, mode='sine_wave', save_name='dataset.pt'):
    """ generate forward movement dataset
    x: action sequence of the first joint in trials
    y: action sequences in trials
    """
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=0.)
    model = Forward(dt=env.dt, seed=seed)
    action_size = env.action_space.shape[0]
    dataset = SimulationDataset(
        data_size, max_episode_steps, 1, action_size, seed, generate_sample,
        # simulation fn kwargs
        env=env, model=model, mode=mode
    )
    print('dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    generate_dataset(data_size=1, max_episode_steps=5000, mode='sine_wave', save_name='forward.pt')
    # generate_dataset(data_size=1, max_episode_steps=5000, mode='square_wave', save_name='forward.pt')
