import torch


class RandomSubset(torch.utils.data.TensorDataset):
    def __init__(self, dataset, data_size, seed):
        """ create a random subset of a TensorDataset """
        x, y = dataset.tensors
        data_size = min(data_size, len(dataset))
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
        indices = indices[:data_size]
        super(RandomSubset, self).__init__(x[indices], y[indices])


class FilterSubset(torch.utils.data.TensorDataset):
    def __init__(self, dataset, data_size):
        """ create a subset of a TensorDataset with its best samples """
        x, y = dataset.tensors
        data_size = min(data_size, len(dataset))
        chemotaxis_index = x.sum(dim=1) / x.size(1)
        indices = torch.argsort(chemotaxis_index, descending=True)
        indices = indices[:data_size]
        super(FilterSubset, self).__init__(x[indices], y[indices])
