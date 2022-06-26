from matplotlib import pyplot as plt
import os
import torch
from virtual_nematode.data.clamp import ClampDataset
from virtual_nematode.data.float import FloatDataset
from virtual_nematode.data.split import SplitDataset
from virtual_nematode.data.subset import random_split, Subset


def plot_data(save_name):
    path = os.path.join('data', save_name)
    dataset = torch.load(path, map_location=torch.device('cpu'))
    x, y = dataset.tensors
    print(x.shape, y.shape)
    plt.subplot(1, 2, 1)
    plt.plot(x[0, :, 0], label='0')
    plt.plot(x[0, :, 12], label='12')
    plt.title('observation')
    plt.subplot(1, 2, 2)
    plt.plot(y[0, :, 0], label='0')
    plt.plot(y[0, :, 48], label='48')
    plt.title('action')
    plt.legend()
    plt.savefig(path + '.png')


def preprocess_dataset(seq_len, load_name, save_name, float_type=False):
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
    if float_type is True:
        dataset = FloatDataset(dataset)
    torch.save(dataset, os.path.join('data', save_name))


def preprocess_and_split_dataset(load_name, save_names, lengths, seq_len, seed):
    # load
    dataset = torch.load(os.path.join('data', load_name))
    print('load', dataset.tensors[0].shape, dataset.tensors[1].shape)
    # subset
    dataset = Subset(dataset, sum(lengths))
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


if __name__ == '__main__':
    # preprocess_dataset(seq_len=32, load_name='data.pt', save_name='data32.pt')
    # preprocess_dataset(seq_len=320, load_name='data.pt', save_name='data320.pt')
    # preprocess_dataset(seq_len=64, load_name='data_640.pt', save_name='data_640_64.pt')
    # preprocess_dataset(seq_len=128, load_name='data_640.pt', save_name='data_640_128.pt')
    # preprocess_dataset(seq_len=64, load_name='data_new_640.pt', save_name='data_new_640_64.pt')
    # preprocess_dataset(seq_len=64, load_name='data_new_640.pt', save_name='data_new_640_64_32bit.pt', float_type=True)
    # plot_data(save_name='data_new_640_64.pt')
    preprocess_and_split_dataset(
        load_name='data_7000_640.pt',
        save_names=[
            'data_5000_640_64_train.pt',
            'data_1000_640_64_eval.pt',
            'data_1000_640_64_test.pt'
        ],
        lengths=[5000, 1000, 1000], seq_len=64, seed=33
    )
