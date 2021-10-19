import torch


class ConcatDataset(torch.utils.data.TensorDataset):
    """ concat datasets """
    def __init__(self, datasets):
        tensors = [dataset.tensors for dataset in datasets]
        x = [x for x, _ in tensors]
        y = [y for _, y in tensors]
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        super(ConcatDataset, self).__init__(x, y)
