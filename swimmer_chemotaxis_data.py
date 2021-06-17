from src.data.chemotaxis import ChemotaxisDataset
from src.models.computational_model import ChemotaxisMotion
from src.utils import clock_position
from swimmer_chemotaxis import make_swimmer
import torch


def generate_chemotaxis_dataset(distance=15, data_size=72000, seed=42, max_episode_steps=2500):
    envs = [make_swimmer(max_episode_steps=max_episode_steps, x=pos_x, y=pos_y) for pos_x, pos_y in clock_position(distance=distance)]
    models = [ChemotaxisMotion(dt=env.dt) for env in envs]
    dataset = ChemotaxisDataset(
        envs=envs,
        models=models,
        data_size=data_size,
        max_episode_steps=max_episode_steps,
        seed=seed
    )
    print('dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, 'data/chemotaxis_dataset.pt')


if __name__ == '__main__':
    generate_chemotaxis_dataset(data_size=72000, max_episode_steps=2500)
