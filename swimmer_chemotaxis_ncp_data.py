""" swimmer: chemotaxis
preprocess chemotaxis dataset for NCP network training
"""

from src.data.ncp import NCPDataset
import torch


def preprocess_dataset():
    dataset = torch.load('data/chemotaxis.pt')
    dataset = NCPDataset(dataset)
    print('dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, 'data/ncp.pt')


if __name__ == '__main__':
    preprocess_dataset()
