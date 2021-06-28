""" swimmer: chemotaxis
preprocess concatenated chemotaxis dataset for NCP network training
"""

from src.data.concat import ConcatDataset
from src.data.ncp import NCPDataset
from src.data.subsequence import SubsequenceDataset
from src.data.subset import FilterSubset
import torch


def preprocess_dataset(data_size=600, seq_len=16):
    """ preprocess chemotaxis dataset with different chemical source positions
    data_size: total amount of original chemotaxis sequences (divided equally between different envs)
    seq_len: split each chemotaxis sequence into subsequences with seq_len
    return: x size [93600, 16, 2], y size [93600, 16, 11]
    """
    # load, filter and concat dataset
    concat_dataset = torch.load('data/concat_chemotaxis.pt')
    data_size = data_size // len(concat_dataset.datasets)
    datasets = [FilterSubset(dataset, data_size) for dataset in concat_dataset.datasets]
    dataset = ConcatDataset(datasets)
    # preprocess dataset
    dataset = NCPDataset(dataset)
    dataset = SubsequenceDataset(dataset, seq_len=seq_len)
    print('dataset', len(concat_dataset), len(dataset), dataset[0][0].size(), dataset[0][1].size())
    torch.save(dataset, 'data/ncp.pt')


if __name__ == '__main__':
    preprocess_dataset(data_size=600, seq_len=16)
