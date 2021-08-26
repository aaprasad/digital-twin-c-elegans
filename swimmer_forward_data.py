""" swimmer: forward sinusoidal movement
generate forward sinusoidal movement dataset
"""

import os
from src.data.simulation import SimulationDataset
from src.models.forward import Forward
from src.utils import sample_seed
from swimmer_forward import make_swimmer
import torch


def generate_sample(env, model, mode):
    """ run a forward movement simulation
    x: torch.Tensor, (max_episode_steps, )
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
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def generate_dataset(data_size=100, seed=42, max_episode_steps=2500, mode='sine_wave', save_name='dataset.pt'):
    """ generate forward movement dataset
    x: action sequence of the first joint in trials
    y: action sequences in trials
    """
    env = make_swimmer(max_episode_steps=max_episode_steps)
    model = Forward(dt=env.dt, seed=seed)
    action_size = env.action_space.shape[0]
    dataset = SimulationDataset(
        data_size, max_episode_steps, action_size, seed, generate_sample,
        # sample fn kwargs
        env=env, model=model, mode=mode
    )
    print('dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    generate_dataset(data_size=100, max_episode_steps=2500, mode='square_wave', save_name='forward.pt')
