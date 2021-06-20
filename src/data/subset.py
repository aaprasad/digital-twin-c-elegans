import torch


class SubsetDataset(torch.utils.data.TensorDataset):
    def __init__(self, dataset, data_size, seed):
        """ create a random subset of a TensorDataset """
        x, y = dataset.tensors
        data_size = min(data_size, len(dataset))
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
        indices = indices[:data_size]
        super(SubsetDataset, self).__init__(x[indices], y[indices])
