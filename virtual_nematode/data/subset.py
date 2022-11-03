import torch


def random_split(dataset, lengths, seed):
    # indices
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    train_indices = indices[:lengths[0]]
    eval_indices = indices[lengths[0]:lengths[0] + lengths[1]]
    test_indices = indices[lengths[0] + lengths[1]:lengths[0] + lengths[1] + lengths[2]]
    # split
    x, y = dataset.tensors
    train_data = torch.utils.data.TensorDataset(x[train_indices], y[train_indices])
    eval_data = torch.utils.data.TensorDataset(x[eval_indices], y[eval_indices])
    test_data = torch.utils.data.TensorDataset(x[test_indices], y[test_indices])
    return train_data, eval_data, test_data


class Subset(torch.utils.data.TensorDataset):
    """ create a subset of a TensorDataset """
    def __init__(self, dataset, data_size, x_index):
        x, y = dataset.tensors
        data_size = min(data_size, len(dataset))
        if type(x_index) is list:
            flag = self.get_index(length=x.shape[2], index=x_index)
            x = x[0:data_size, :, flag]
        else:
            x = x[0:data_size, :, x_index[0]:x_index[1]]
        y = y[0:data_size]
        super(Subset, self).__init__(x, y)

    @staticmethod
    def get_index(length, index):
        flag = torch.zeros(length, dtype=torch.bool)
        for a, b in index:
            flag[a:b] = True
        return flag


class RandomSubset(torch.utils.data.TensorDataset):
    """ create a random subset of a TensorDataset """
    def __init__(self, dataset, data_size, seed):
        x, y = dataset.tensors
        data_size = min(data_size, len(dataset))
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
        indices = indices[:data_size]
        super(RandomSubset, self).__init__(x[indices], y[indices])


class FilterSubset(torch.utils.data.TensorDataset):
    """ create a subset of a TensorDataset with its best samples according to chemotaxis index """
    def __init__(self, dataset, data_size):
        x, y = dataset.tensors
        data_size = min(data_size, len(dataset))
        chemotaxis_index = x.squeeze(dim=2).sum(dim=1) / x.size(1)
        indices = torch.argsort(chemotaxis_index, descending=True)
        indices = indices[:data_size]
        super(FilterSubset, self).__init__(x[indices], y[indices])
