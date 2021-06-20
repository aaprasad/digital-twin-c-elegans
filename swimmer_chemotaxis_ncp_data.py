""" swimmer: chemotaxis
preprocess concatenated chemotaxis dataset for NCP network training
"""

from src.data.ncp import NCPDataset
from src.data.subset import RandomSubset
import torch
from tqdm import tqdm


def preprocess_dataset(data_size=12000, seed=42):
    concat_dataset = torch.load('data/concat_chemotaxis.pt')
    data_size = data_size // len(concat_dataset.datasets)
    datasets = []
    for dataset in tqdm(concat_dataset.datasets):
        dataset = RandomSubset(dataset, data_size=data_size, seed=seed)
        dataset = NCPDataset(dataset)
        datasets.append(dataset)
    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    print('dataset', len(concat_dataset), concat_dataset[0][0].size(), concat_dataset[0][1].size())
    torch.save(concat_dataset, 'data/concat_ncp.pt')


if __name__ == '__main__':
    preprocess_dataset(data_size=12000)
