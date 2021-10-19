""" swimmer: open-loop control of forward locomotion
preprocess forward simulation dataset for NCP network training
"""

import os
import torch
from virtual_nematode.data.split import SplitDataset
from virtual_nematode.data.subset import RandomSubset


def preprocess_dataset(data_size=50, seed=42, seq_len=16, load_name='source.pt', save_name='target.pt'):
    # load dataset
    dataset = torch.load(os.path.join('data', load_name))
    print('load dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    # filter dataset
    dataset = RandomSubset(dataset, data_size, seed)
    print('random subset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    # split dataset
    dataset = SplitDataset(dataset, seq_len=seq_len)
    print('split dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    preprocess_dataset(data_size=1, seq_len=16, load_name='data.pt', save_name='ncp.pt')
