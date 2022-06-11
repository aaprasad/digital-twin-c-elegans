import torch


class ClampDataset(torch.utils.data.TensorDataset):
    def __init__(self, dataset, x_range=None, y_range=None):
        x, y = dataset.tensors
        if x_range is not None:
            x = torch.clamp(x, min=x_range[0], max=x_range[1])
        if y_range is not None:
            y = torch.clamp(y, min=y_range[0], max=y_range[1])
        super(ClampDataset, self).__init__(x, y)
