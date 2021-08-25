""" swimmer: forward
preprocess forward simulation dataset for NCP network training
"""

import os
from src.data.split import SplitDataset
from src.data.subset import FilterSubset
import torch


def preprocess_dataset(data_size=50, seq_len=16, load_name='source.pt', save_name='target.pt'):
    # load dataset
    dataset = torch.load(os.path.join('data', load_name))
    print('loaded dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    # filter dataset
    dataset = FilterSubset(dataset, data_size)
    print('filtered dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    # preprocess dataset
    dataset = SplitDataset(dataset, seq_len=seq_len)
    print('processed dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    preprocess_dataset(data_size=50, seq_len=16, load_name='forward.pt', save_name='forward_ncp.pt')
