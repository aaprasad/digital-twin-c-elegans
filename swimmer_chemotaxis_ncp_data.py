""" swimmer: chemotaxis
preprocess chemotaxis dataset for NCP network training
"""

from src.data.ncp import NCPDataset
from src.data.subset import SubsetDataset
import torch


def preprocess_dataset(data_size=60000, seed=42):
    dataset = torch.load('data/chemotaxis.pt')
    dataset = SubsetDataset(dataset, data_size=data_size, seed=seed)
    dataset = NCPDataset(dataset)
    print('dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, 'data/ncp.pt')


if __name__ == '__main__':
    preprocess_dataset(data_size=60000)
