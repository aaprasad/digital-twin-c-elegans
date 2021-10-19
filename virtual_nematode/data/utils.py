import multiprocessing
import os
import torch
from virtual_nematode.data.split import SplitDataset
from virtual_nematode.data.subset import RandomSubset


def prepare_dataloader(data_path, eval_ratio, test_ratio, batch_size, seed):
    """ load data and prepare data loaders """
    dataset = torch.load(data_path)
    eval_size = int(len(dataset) * eval_ratio)
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - eval_size - test_size
    train_data, eval_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, eval_size, test_size], generator=torch.Generator().manual_seed(seed)
    )
    kwargs = {'drop_last': False, 'num_workers': multiprocessing.cpu_count(), 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
    print('dataset', len(dataset), [len(train_loader.dataset), len(eval_loader.dataset), len(test_loader.dataset)])
    return train_loader, eval_loader, test_loader


def subset_and_split(data_size=50, seed=42, seq_len=16, load_name='source.pt', save_name='target.pt'):
    # load dataset
    dataset = torch.load(os.path.join('data', load_name))
    print('loaded dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    # filter dataset
    dataset = RandomSubset(dataset, data_size, seed)
    print('random subset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    # preprocess dataset
    dataset = SplitDataset(dataset, seq_len=seq_len)
    print('processed dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, os.path.join('data', save_name))
