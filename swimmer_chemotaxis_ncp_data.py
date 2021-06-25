""" swimmer: chemotaxis
preprocess concatenated chemotaxis dataset for NCP network training
"""

from src.data.subsequence import SubsequenceDataset
from src.data.encode import EncodeDataset
from src.data.subset import FilterSubset
import torch
from tqdm import tqdm


def preprocess_dataset(data_size=600, seq_len=16):
    """ preprocess chemotaxis dataset with different chemical source positions
    data_size: total amount of original chemotaxis sequences (divided equally between different envs)
    seq_len: split each chemotaxis sequence into subsequences with seq_len
    return: x size [93600, 16, 2], y size [93600, 16, 11]
    """
    concat_dataset = torch.load('data/concat_chemotaxis.pt')
    data_size = data_size // len(concat_dataset.datasets)
    datasets = []
    for dataset in tqdm(concat_dataset.datasets):
        dataset = FilterSubset(dataset, data_size=data_size)
        dataset = EncodeDataset(dataset)
        dataset = SubsequenceDataset(dataset, seq_len=seq_len)
        datasets.append(dataset)
    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    print('dataset', len(concat_dataset), concat_dataset[0][0].size(), concat_dataset[0][1].size())
    torch.save(concat_dataset, 'data/concat_ncp.pt')


if __name__ == '__main__':
    preprocess_dataset(data_size=600, seq_len=16)
