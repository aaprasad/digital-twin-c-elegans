import torch


class Sample(torch.utils.data.Dataset):
    """ run a simulation as a sample """
    def __init__(self, data_size, get_item, **kwargs):
        super(Sample, self).__init__()
        self.data_size = data_size
        self.get_item = get_item  # fn of simulation
        self.kwargs = kwargs  # kwargs for simulation

    def __getitem__(self, index):
        return self.get_item(**self.kwargs)

    def __len__(self):
        return self.data_size
