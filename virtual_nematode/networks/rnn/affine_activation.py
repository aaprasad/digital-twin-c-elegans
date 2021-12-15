import torch


class AffineActivation(torch.nn.Module):
    def __init__(self, input_size):
        super(AffineActivation, self).__init__()
        self.w = torch.nn.Parameter(torch.ones(input_size))
        self.b = torch.nn.Parameter(torch.zeros(input_size))

    def forward(self, input):
        return input * self.w + self.b
