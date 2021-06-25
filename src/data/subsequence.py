import torch


class SubsequenceDataset(torch.utils.data.TensorDataset):
    """ split each sequence into subsequences """
    def __init__(self, dataset, seq_len):
        self.seq_len = seq_len
        x, y = dataset.tensors
        x = self.subsequence(x)
        y = self.subsequence(y)
        super(SubsequenceDataset, self).__init__(x, y)

    def subsequence(self, tensor):
        tensor = tensor.split(self.seq_len, dim=1)
        if tensor[-1].size(1) != self.seq_len:
            tensor = tensor[:-1]
        tensor = torch.cat(tensor, dim=0)
        return tensor
