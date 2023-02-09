import math
import torch


def muscle_band_arrangement(band_length, kernel_size):
    mask = torch.zeros((band_length, band_length), dtype=torch.bool)
    kernel_width = int(kernel_size / 2)
    for i in range(band_length):
        mask[max(i-kernel_width, 0):min(i+kernel_width+1, band_length), i] = True
    init_range = 1 / math.sqrt(kernel_size)
    weight = torch.zeros((band_length, band_length)).uniform_(-init_range, init_range)
    bias = torch.zeros(kernel_size).uniform_(-init_range, init_range)
    return weight, bias, mask


class MuscleArrangement(torch.nn.Module):
    def __init__(self, band_lengths, kernel_size):
        super(MuscleArrangement, self).__init__()
        weight, bias, mask = [], [], []
        for band_length in band_lengths:
            w, b, m = muscle_band_arrangement(band_length, kernel_size)
            w = torch.nn.Parameter(w)
            b = torch.nn.Parameter(b)
            m = torch.nn.Parameter(m, requires_grad=False)
            weight.append(w)
            bias.append(b)
            mask.append(m)
        self.band_lengths = band_lengths
        self.band_range = torch.cumsum(torch.tensor([0] + band_lengths), dim=0)
        self.weight = weight
        self.bias = bias
        self.mask = mask

    def forward(self, action):  # (batch_size, m)
        action_split = [action[:, self.band_range[i]:self.band_range[i+1]] for i in range(len(self.band_lengths))]
        action_split = [
            torch.mm(a, w * m) + b for a, w, b, m in zip(action_split, self.weight, self.bias, self.mask)
        ]
        action = torch.cat(action_split, dim=1)
        action = action.clamp(0, 1)
        return action
