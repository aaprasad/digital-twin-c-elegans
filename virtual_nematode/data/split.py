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
