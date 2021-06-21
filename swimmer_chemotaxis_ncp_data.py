""" swimmer: chemotaxis
preprocess concatenated chemotaxis dataset for NCP network training
"""

from src.data.encode import EncodeDataset
from src.data.subset import FilterSubset
import torch
from tqdm import tqdm


def preprocess_dataset(data_size=12000):
    """ preprocess chemotaxis dataset with different chemical source positions
    data_size: the total dataset size, should be divided for each env (with different source position)
    """
    concat_dataset = torch.load('data/concat_chemotaxis.pt')
    data_size = data_size // len(concat_dataset.datasets)
    datasets = []
    for dataset in tqdm(concat_dataset.datasets):
        dataset = FilterSubset(dataset, data_size=data_size)
        dataset = EncodeDataset(dataset)
        datasets.append(dataset)
    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    print('dataset', len(concat_dataset), concat_dataset[0][0].size(), concat_dataset[0][1].size())
    torch.save(concat_dataset, 'data/concat_ncp.pt')


if __name__ == '__main__':
    preprocess_dataset(data_size=12000)
