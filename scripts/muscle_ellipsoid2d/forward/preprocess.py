import os
import torch
from virtual_nematode.data.clamp import ClampDataset
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
    dataset = ClampDataset(dataset, x_range=None, y_range=[0, 1])
    print(
        'clamp dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size(),
        'x range', dataset.tensors[0].min().item(), dataset.tensors[0].max().item(),
        'y range', dataset.tensors[1].min().item(), dataset.tensors[1].max().item()
    )
    torch.save(dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    # preprocess_dataset(seq_len=32, load_name='data.pt', save_name='data32.pt')
    # preprocess_dataset(seq_len=320, load_name='data.pt', save_name='data320.pt')
    # preprocess_dataset(seq_len=64, load_name='data_640.pt', save_name='data_640_64.pt')
    # preprocess_dataset(seq_len=128, load_name='data_640.pt', save_name='data_640_128.pt')
    preprocess_dataset(seq_len=64, load_name='data_new_640.pt', save_name='data_new_640_64.pt')
