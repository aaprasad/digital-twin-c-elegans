""" swimmer: chemotaxis
generate concatenated chemotaxis dataset
"""

from src.data.chemotaxis import ChemotaxisDataset
from src.models.chemotaxis_motion import ChemotaxisMotion
from src.utils import clock_position
from swimmer_chemotaxis import make_swimmer
import torch


def generate_dataset(distance=15, data_size=12000, seed=42, max_episode_steps=2500):
    """ generate chemotaxis dataset with different chemical source positions
    data_size: the total dataset size, should be divided for each env (with different source position)
    """
    envs = [make_swimmer(max_episode_steps=max_episode_steps, x=x, y=y) for x, y in clock_position(distance=distance)]
    models = [ChemotaxisMotion(dt=env.dt) for env in envs]
    data_size = data_size // len(envs)
    datasets = []
    for env, model in zip(envs, models):
        dataset = ChemotaxisDataset(env, model, data_size=data_size, max_episode_steps=max_episode_steps, seed=seed)
        print(dataset.source, len(dataset), dataset[0][0].size(), dataset[0][1].size())
        datasets.append(dataset)
    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    print('dataset', len(concat_dataset), concat_dataset[0][0].size(), concat_dataset[0][1].size())
    torch.save(concat_dataset, 'data/concat_chemotaxis.pt')


if __name__ == '__main__':
    generate_dataset(data_size=12000, max_episode_steps=2500)
