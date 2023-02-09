import math
import torch


def muscle_band_arrangement(band_length, kernel_size):
    mask = torch.zeros((band_length, band_length), dtype=torch.bool)
    kernel_width = int(kernel_size / 2)
    for i in range(band_length):
        mask[max(i-kernel_width, 0):min(i+kernel_width+1, band_length), i] = True
    init_range = 1 / math.sqrt(kernel_size)
    weight = torch.zeros((band_length, band_length)).uniform_(-init_range, init_range)
    bias = torch.zeros(band_length).uniform_(-init_range, init_range)
    return weight, bias, mask


def muscle_band_arrangement1(band_length, kernel_size):
    mask = torch.zeros((band_length, band_length), dtype=torch.bool)
    kernel_width = int(kernel_size / 2)
    for i in range(band_length):
        mask[max(i-kernel_width, 0):min(i+kernel_width+1, band_length), i] = True
    init_range = 1 / kernel_size
    weight = torch.zeros((band_length, band_length)).uniform_(-init_range, init_range)
    bias = torch.zeros(band_length).uniform_(-init_range, init_range)
    return weight, bias, mask


def muscle_band_arrangement2(band_length, kernel_size):
    mask = torch.zeros((band_length, band_length), dtype=torch.bool)
    kernel_width = int(kernel_size / 2)
    for i in range(band_length):
        mask[max(i-kernel_width, 0):min(i+kernel_width+1, band_length), i] = True
    init_range = 1 / kernel_size
    weight = torch.zeros((band_length, band_length)).uniform_(0, init_range)
    bias = torch.zeros(band_length).uniform_(0, init_range)
    return weight, bias, mask


class MuscleArrangement(torch.nn.Module):
    def __init__(self, band_lengths, kernel_size):
        super(MuscleArrangement, self).__init__()
        self.band_lengths = band_lengths
        self.band_range = torch.cumsum(torch.tensor([0] + band_lengths), dim=0)
        # band 0
        w0, b0, m0 = muscle_band_arrangement2(band_lengths[0], kernel_size)
        self.w0 = torch.nn.Parameter(w0)
        self.b0 = torch.nn.Parameter(b0)
        self.m0 = torch.nn.Parameter(m0, requires_grad=False)
        # band 1
        w1, b1, m1 = muscle_band_arrangement(band_lengths[1], kernel_size)
        self.w1 = torch.nn.Parameter(w1)
        self.b1 = torch.nn.Parameter(b1)
        self.m1 = torch.nn.Parameter(m1, requires_grad=False)
        # band 2
        w2, b2, m2 = muscle_band_arrangement(band_lengths[2], kernel_size)
        self.w2 = torch.nn.Parameter(w2)
        self.b2 = torch.nn.Parameter(b2)
        self.m2 = torch.nn.Parameter(m2, requires_grad=False)
        # band 3
        w3, b3, m3 = muscle_band_arrangement(band_lengths[3], kernel_size)
        self.w3 = torch.nn.Parameter(w3)
        self.b3 = torch.nn.Parameter(b3)
        self.m3 = torch.nn.Parameter(m3, requires_grad=False)

    def forward(self, action):  # (batch_size, m)
        action_split = [action[:, self.band_range[i]:self.band_range[i+1]] for i in range(len(self.band_lengths))]
        action_split = [
            torch.mm(action_split[0], self.w0 * self.m0), #+ self.b0,
            torch.mm(action_split[1], self.w1 * self.m1), #+ self.b1,
            torch.mm(action_split[2], self.w2 * self.m2), #+ self.b2,
            torch.mm(action_split[3], self.w3 * self.m3), #+ self.b3
        ]
        action = torch.cat(action_split, dim=1)
        action = action.clamp(0, 1)
        return action
