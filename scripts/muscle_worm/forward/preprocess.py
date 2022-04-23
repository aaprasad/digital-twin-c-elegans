import os
import torch
from virtual_nematode.data.split import SplitDataset


def preprocess_dataset(seq_len, load_name, save_name):
    """
    x: torch.Tensor, (data_size, seq_len, input_size)
    y: torch.Tensor, (data_size, seq_len, action_size)
    """
    dataset = torch.load(os.path.join('data', load_name))
    print('load dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    dataset = SplitDataset(dataset, seq_len=seq_len)
    print('split dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    preprocess_dataset(seq_len=16, load_name='data.pt', save_name='ncp.pt')
