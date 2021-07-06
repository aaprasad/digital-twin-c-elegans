""" swimmer: chemotaxis
preprocess concatenated chemotaxis dataset for NCP network training
"""

import numpy as np
import os
from src.data.concat import ConcatDataset
from src.data.ncp import NCPDataset
from src.data.subset import FilterSubset
import torch


def preprocess_dataset(data_size=600, seq_len=16, load_name='concat_chemotaxis.pt', save_name='ncp.pt'):
    """ preprocess chemotaxis dataset with different chemical source positions
    data_size: total amount of original chemotaxis sequences (divided equally between different envs)
    seq_len: split each chemotaxis sequence into subsequences with seq_len
    return: x size [93600, 16, 2], y size [93600, 16, 11]
    """
    # load dataset
    concat_dataset = torch.load(os.path.join('data', load_name))
    print('loaded dataset', len(concat_dataset), concat_dataset[0][0].size(), concat_dataset[0][1].size())
    print('chemotaxis index mean', np.mean([torch.mean(dataset.tensors[0]).item() for dataset in concat_dataset.datasets]))
    # filter and concat dataset
    data_size = data_size // len(concat_dataset.datasets)
    datasets = [FilterSubset(dataset, data_size) for dataset in concat_dataset.datasets]
    dataset = ConcatDataset(datasets)
    print('filtered dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    print('chemotaxis index mean', torch.mean(dataset.tensors[0]).item())
    # preprocess dataset
    dataset = NCPDataset(dataset, seq_len=seq_len)
    print('processed dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    preprocess_dataset(data_size=600, seq_len=16, load_name='concat_chemotaxis.pt', save_name='ncp.pt')
