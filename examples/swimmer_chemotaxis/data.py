""" swimmer: chemotaxis
generate concatenated chemotaxis dataset
"""

import os
from virtual_nematode.data.simulation import SimulationDataset
from virtual_nematode.models.computational_model import ComputationalModelChemotaxis
from virtual_nematode.utils import clock_position, sample_seed
from sim import make_swimmer
import torch


def generate_sample(env, model):
    """ run a chemotaxis simulation controlled by strategies
    x: torch.Tensor, (max_episode_steps, 1)
    y: torch.Tensor, (max_episode_steps, action_size)
    """
    seed = sample_seed()
    env.seed(seed)  # seed env
    model.seed(seed)  # seed model
    observation = env.reset()
    info = env.get_info(info={})
    y = []
    for i in range(10 ** 6):
        # env.render()
        action = model.step(step=i, q=observation[1:12], q_vel=observation[15:], g_p=info['g_p'], g_w=info['g_w'])
        y.append(action.tolist())
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    x = torch.tensor(env.stats['concentration'], dtype=torch.float64).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float64)
    return x, y


def generate_dataset(distance=15, data_size=12000, seed=42, max_episode_steps=2500, save_name='dataset.pt'):
    """ generate chemotaxis dataset with different chemical source positions
    x: concentrations sensed at nose tip
    y: actions performed each step
    data_size: the total dataset size, should be divided for each env (with different source position)
    seed: the randomly generated dataset stays the same with seeding
    """
    envs = [make_swimmer(max_episode_steps=max_episode_steps, x=x, y=y) for x, y in clock_position(distance=distance)]
    models = [ComputationalModelChemotaxis(dt=env.dt) for env in envs]
    data_size = data_size // len(envs)
    datasets = []
    for env, model in zip(envs, models):
        action_size = env.action_space.shape[0]
        dataset = SimulationDataset(
            data_size, max_episode_steps, 1, action_size, seed, generate_sample, env=env, model=model
        )
        print(env.source.tolist(), len(dataset), dataset[0][0].size(), dataset[0][1].size())
        datasets.append(dataset)
    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    print('dataset', len(concat_dataset), concat_dataset[0][0].size(), concat_dataset[0][1].size())
    os.makedirs('data', exist_ok=True)
    torch.save(concat_dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    generate_dataset(data_size=12000, max_episode_steps=2500, save_name='computational_model_chemotaxis.pt')
