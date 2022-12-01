import multiprocessing
import os
import torch
from virtual_nematode.data.clamp import ClampDataset
from virtual_nematode.data.float import FloatDataset
from virtual_nematode.data.split import SplitDataset, SlidingWindowDataset
from virtual_nematode.data.subset import random_split, Subset


def preprocess_and_split_dataset(load_name, save_names, lengths, x_index, seq_len, seed):
    # load
    dataset = torch.load(os.path.join('data', load_name))
    print('load', dataset.tensors[0].shape, dataset.tensors[1].shape)
    # subset
    dataset = Subset(dataset, sum(lengths), x_index)
    print('subset', dataset.tensors[0].shape, dataset.tensors[1].shape)
    # float
    dataset = FloatDataset(dataset)
    print('dtype', dataset.tensors[0].dtype, dataset.tensors[1].dtype)
    # clamp
    dataset = ClampDataset(dataset, x_range=None, y_range=[0, 1])
    print(
        'clamp',
        'x range', dataset.tensors[0].min().item(), dataset.tensors[0].max().item(),
        'y range', dataset.tensors[1].min().item(), dataset.tensors[1].max().item()
    )
    # split train, eval, test
    train_data, eval_data, test_data = random_split(dataset, lengths, seed)
    print('train', train_data.tensors[0].shape, train_data.tensors[1].shape)
    print('eval', eval_data.tensors[0].shape, eval_data.tensors[1].shape)
    print('test', test_data.tensors[0].shape, test_data.tensors[1].shape)
    # split sequence
    train_data = SplitDataset(train_data, seq_len)
    print('split sequence train', train_data.tensors[0].shape, train_data.tensors[1].shape)
    eval_data = SplitDataset(eval_data, seq_len)
    print('split sequence eval', eval_data.tensors[0].shape, eval_data.tensors[1].shape)
    test_data = SplitDataset(test_data, seq_len)
    print('split sequence test', test_data.tensors[0].shape, test_data.tensors[1].shape)
    # save
    torch.save(train_data, os.path.join('data', save_names[0]))
    torch.save(eval_data, os.path.join('data', save_names[1]))
    torch.save(test_data, os.path.join('data', save_names[2]))


def preprocess_and_split_slide_dataset(load_name, save_names, lengths, x_index, seq_len, seq_amount, stride, seed):
    # load
    dataset = torch.load(os.path.join('data', load_name))
    print('load', dataset.tensors[0].shape, dataset.tensors[1].shape)
    # subset
    dataset = Subset(dataset, sum(lengths), x_index)
    print('subset', dataset.tensors[0].shape, dataset.tensors[1].shape)
    # float
    dataset = FloatDataset(dataset)
    print('dtype', dataset.tensors[0].dtype, dataset.tensors[1].dtype)
    # clamp
    dataset = ClampDataset(dataset, x_range=None, y_range=[0, 1])
    print(
        'clamp',
        'x range', dataset.tensors[0].min().item(), dataset.tensors[0].max().item(),
        'y range', dataset.tensors[1].min().item(), dataset.tensors[1].max().item()
    )
    # split train, eval, test
    train_data, eval_data, test_data = random_split(dataset, lengths, seed)
    print('train', train_data.tensors[0].shape, train_data.tensors[1].shape)
    print('eval', eval_data.tensors[0].shape, eval_data.tensors[1].shape)
    print('test', test_data.tensors[0].shape, test_data.tensors[1].shape)
    # split sequence
    train_data = SlidingWindowDataset(train_data, seq_len, seq_amount, stride)
    print('split sequence train', train_data.tensors[0].shape, train_data.tensors[1].shape)
    eval_data = SlidingWindowDataset(eval_data, seq_len, seq_amount, stride)
    print('split sequence eval', eval_data.tensors[0].shape, eval_data.tensors[1].shape)
    test_data = SlidingWindowDataset(test_data, seq_len, seq_amount, stride)
    print('split sequence test', test_data.tensors[0].shape, test_data.tensors[1].shape)
    # save
    torch.save(train_data, os.path.join('data', save_names[0]))
    torch.save(eval_data, os.path.join('data', save_names[1]))
    torch.save(test_data, os.path.join('data', save_names[2]))


def preprocess_single_dataset(load_name, save_name, data_size, x_index):
    # load
    dataset = torch.load(os.path.join('data', load_name))
    print('load', dataset.tensors[0].shape, dataset.tensors[1].shape)
    # subset
    dataset = Subset(dataset, data_size, x_index)
    print('subset', dataset.tensors[0].shape, dataset.tensors[1].shape)
    # float
    dataset = FloatDataset(dataset)
    print('dtype', dataset.tensors[0].dtype, dataset.tensors[1].dtype)
    # clamp
    dataset = ClampDataset(dataset, x_range=None, y_range=[0, 1])
    print(
        'clamp',
        'x range', dataset.tensors[0].min().item(), dataset.tensors[0].max().item(),
        'y range', dataset.tensors[1].min().item(), dataset.tensors[1].max().item()
    )
    # save
    torch.save(dataset, os.path.join('data', save_name))


def prepare_dataloader(data_path, batch_size):
    """ load data and prepare data loaders """
    train_data = torch.load(data_path[0])
    eval_data = torch.load(data_path[1])
    test_data = torch.load(data_path[2])
    kwargs = {'drop_last': False, 'num_workers': multiprocessing.cpu_count(), 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
    print('dataset', [len(train_loader.dataset), len(eval_loader.dataset), len(test_loader.dataset)])
    return train_loader, eval_loader, test_loader


def prepare_test_dataloader(data_path, batch_size):
    test_data = torch.load(data_path)
    kwargs = {'drop_last': False, 'num_workers': multiprocessing.cpu_count(), 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
    print('dataset', len(test_loader.dataset))
    return test_loader
