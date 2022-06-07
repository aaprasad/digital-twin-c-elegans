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
