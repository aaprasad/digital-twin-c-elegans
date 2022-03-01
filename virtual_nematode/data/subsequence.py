import torch


class SubsequenceDataset(torch.utils.data.TensorDataset):
    def __init__(self, dataset, seq_range):
        x, y = dataset.tensors
        start, end = seq_range
        x = x[:, start:end, :]
        y = y[:, start:end, :]
        super(SubsequenceDataset, self).__init__(x, y)
