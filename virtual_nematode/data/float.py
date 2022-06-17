import torch


class FloatDataset(torch.utils.data.TensorDataset):
    def __init__(self, dataset):
        x, y = dataset.tensors
        super(FloatDataset, self).__init__(x.float(), y.float())
