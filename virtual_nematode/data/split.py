import torch


class SplitDataset(torch.utils.data.TensorDataset):
    """ split sequence in dataset
    x: torch.Tensor, (data_size, seq_len, 1)
    y: torch.Tensor, (data_size, seq_len, action_size)
    """
    def __init__(self, dataset, seq_len):
        x, y = dataset.tensors
        self.seq_len = seq_len
        x = self.subsequence(x)
        y = self.subsequence(y)
        super(SplitDataset, self).__init__(x, y)

    def subsequence(self, tensor):
        """ split each sequence into subsequences
        target seq_len must NOT be larger than original seq_len
        """
        seq_len = min(self.seq_len, tensor.size(1))
        tensor = tensor.split(seq_len, dim=1)
        if tensor[-1].size(1) != seq_len:
            tensor = tensor[:-1]
        tensor = torch.cat(tensor, dim=0)
        return tensor


class SlidingWindowDataset(torch.utils.data.TensorDataset):
    """ split sequence in dataset
    x: torch.Tensor, (data_size, seq_len, 1)
    y: torch.Tensor, (data_size, seq_len, action_size)
    """
    def __init__(self, dataset, seq_len, seq_amount, stride):
        x, y = dataset.tensors
        self.seq_len = seq_len
        self.seq_amount = seq_amount
        self.stride = stride
        x = self.subsequence(x)
        y = self.subsequence(y)
        super(SlidingWindowDataset, self).__init__(x, y)

    def subsequence(self, tensor):
        """ split each sequence into subsequences """
        tensor = [tensor[:, i*self.stride:i*self.stride+self.seq_len, :] for i in range(self.seq_amount)]
        tensor = torch.cat(tensor, dim=0)
        return tensor


class SlidingWindowSampleDataset(torch.utils.data.TensorDataset):
    """ split sequence in dataset
    x: torch.Tensor, (data_size, seq_len, 1)
    y: torch.Tensor, (data_size, seq_len, action_size)
    """
    def __init__(self, dataset, seq_len, seq_range, seq_amount, stride, seed):
        x, y = dataset.tensors
        self.seq_len = seq_len
        self.seq_amount = seq_amount
        self.seq_range = seq_range
        self.stride = stride
        x = self.subsequence(x)
        y = self.subsequence(y)
        assert x.shape[0] == y.shape[0]
        index = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))
        print('index full', index.shape)
        index = index[0:len(dataset)*seq_amount]
        print('index selected', index.shape)
        x, y = x[index, :, :], y[index, :, :]
        super(SlidingWindowSampleDataset, self).__init__(x, y)

    def subsequence(self, tensor):
        seq_range = min(self.seq_range, tensor.shape[1])
        seq_index_range = int((seq_range - self.seq_len) / self.stride) + 1
        tensor = [tensor[:, i*self.stride:i*self.stride+self.seq_len, :] for i in range(seq_index_range)]
        tensor = torch.cat(tensor, dim=0)
        return tensor
