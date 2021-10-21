import os
import torch
from virtual_nematode.data.split import SplitDataset


def preprocess_dataset(seq_len=16, load_name='source.pt', save_name='target.pt'):
    # load dataset
    dataset = torch.load(os.path.join('data', load_name))
    print('load dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    # split dataset
    dataset = SplitDataset(dataset, seq_len=seq_len)
    print('split dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    """ results
    dataset: x [72000, 16, 23], y [72000, 16, 11]
    """
    preprocess_dataset(seq_len=16, load_name='data.pt', save_name='ncp.pt')
