import numpy as np
import pandas as pd
import torch


class Connectome(object):
    def __init__(self, path):
        self.chemical = self._load(path, sheet_name='hermaphrodite chemical')
        self.gap_junction = self._load(path, sheet_name='hermaphrodite gap jn symmetric')

    @staticmethod
    def _load(path, sheet_name):
        df = pd.read_excel(path, sheet_name=sheet_name, header=2, index_col=2).iloc[:-1, 2:-1]
        return df

    def check(self, cells):
        for cell in cells:
            if cell not in self.chemical.index or cell not in self.chemical.columns:
                raise KeyError('Chemical adjacency matrix does not include {}'.format(cell))
            if cell not in self.gap_junction.index or cell not in self.gap_junction.columns:
                raise KeyError('Gap junction adjacency matrix does not include {}'.format(cell))

    def slice(self, cells):
        self.chemical = self.chemical.loc[cells, cells]
        self.gap_junction = self.chemical.loc[cells, cells]

    def weight(self):
        chemical = self.chemical.replace(np.nan, 0).to_numpy(dtype=np.int32)
        gap_junction = self.gap_junction.replace(np.nan, 0).to_numpy(dtype=np.int32)
        return chemical, gap_junction


class SNNCell(torch.nn.Module):
    """ neuronal network model
    https://doi.org/10.1038/s41598-021-92690-2
    """
    def __init__(self, freq, n, p):
        super(SNNCell, self).__init__()
        self.freq = freq  # freq of data sequence
        self.n = n  # cell count
        self.p = p  # proprioception size
        self.tau = torch.nn.Parameter(torch.rand(n))  # cell time constant, (cell_count, )
        self.w_c = torch.nn.Parameter(torch.rand((n, n)))  # chemical synapse weight, (cell_count, cell_count)
        self.w_g = torch.nn.Parameter(torch.rand((n, n)))  # gap junction weight, (cell_count, cell_count)
        self.w_p = torch.nn.Parameter(torch.rand((p, n)))  # proprioception input synapse weight, (proprioception_size, cell_count)
        self.bias = torch.nn.Parameter(torch.rand(n))  # cell state bias, (cell_count, )

    def forward(self, state, activation, proprioception):
        """ forward 1 step
        state: cell state, (batch_size, cell_count)
        activation: cell activation, (batch_size, cell_count)
        proprioception: (batch_size, proprioception_size)
        """
        synapse_input = torch.mm(activation, self.w_c)  # chemical synapse input, (batch_size, cell_count)
        delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
        gap_input = torch.sum(delta_state * self.w_g, dim=1)  # gap junction input, (batch_size, cell_count)
        proprioception_input = torch.mm(proprioception, self.w_p)  # proprioception input, (batch_size, cell_count)
        total_input = synapse_input + gap_input + proprioception_input + self.bias  # total input, (batch_size, cell_count)
        # cell state, (batch_size, cell_count)
        state = 1 / (1 + self.freq * self.tau) * state + self.freq * self.tau / (1 + self.freq * self.tau) * total_input
        # cell activation, (batch_size, cell_count)
        activation = torch.nn.functional.sigmoid(state)
        return state, activation


class SNN(torch.nn.Module):
    def __init__(self, cell):
        super(SNN, self).__init__()
        self.cell = cell

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        state = torch.zeros((batch_size, self.cell.state_size), device=device)
        output = []
        for t in range(seq_len):
            inputs = x[:, t]
            new_output, state = self.cell.forward(inputs, state)
            output.append(new_output)
        output = torch.stack(output, dim=1)  # return entire sequence
        return output

    def step(self, x, state=None):
        if state is None:
            batch_size = x.size(0)
            state = torch.zeros((batch_size, self.cell.state_size))
        output, state = self.cell.forward(x, state)
        return output, state
