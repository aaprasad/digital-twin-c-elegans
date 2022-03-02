import os
import torch
from virtual_nematode.data.chemotaxis import GradientDataset
from virtual_nematode.data.concat import ConcatDataset
from virtual_nematode.data.split import SplitDataset
from virtual_nematode.data.subsequence import SubsequenceDataset
from virtual_nematode.data.subset import RandomSubset
from virtual_nematode.envs.swimmer import fick


def preprocess_dataset(seq_ranges, data_size, seq_len=16, seed=11, load_name='source.pt', save_name='target.pt'):
    """
    x: torch.Tensor, (data_size, seq_len, input_size)
    y: torch.Tensor, (data_size, seq_len, action_size)
    """
    dataset = torch.load(os.path.join('data', load_name))
    print('load dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    datasets = []
    for i, seq_range in enumerate(seq_ranges):
        d = SubsequenceDataset(dataset, seq_range)
        print(i, 'subsequence dataset', len(d), d[0][0].size(), d[0][1].size())
        d = SplitDataset(d, seq_len)
        print(i, 'split dataset', len(d), d[0][0].size(), d[0][1].size())
        datasets.append(d)
    dataset = ConcatDataset(datasets)
    print('concat dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    dataset = RandomSubset(dataset, data_size, seed)
    print('random subset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    dataset = GradientDataset(dataset, p_range=(0, 48), g_index=48, g_coef=fick())
    print('gradient dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size(), 'max abs gradient', fick())
    torch.save(dataset, os.path.join('data', save_name))


if __name__ == '__main__':
    preprocess_dataset([(0, 400)], data_size=72000, seq_len=16, seed=11, load_name='data.pt', save_name='steering.pt')
    # preprocess_dataset([(0, 3500)], data_size=72000, seq_len=16, seed=11, load_name='data.pt', save_name='full.pt')
