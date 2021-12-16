import torch


class Identity(torch.nn.Module):
    def __init__(self, output_size):
        super(Identity, self).__init__()
        self.output_size = output_size

    def forward(self, input):
        return input[:, 0:self.output_size]
