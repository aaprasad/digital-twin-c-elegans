import numpy as np
import pandas as pd
import torch


class Connectome(object):
    def __init__(self, cells, ex_synapses, in_synapses, path):
        self.cells = cells
        self.ex_synapses = ex_synapses
        self.in_synapses = in_synapses
        self.chemical = pd.read_excel(path, sheet_name='hermaphrodite chemical', header=2, index_col=2).iloc[:300, 2:456]
        self.gap_junction = pd.read_excel(path, sheet_name='hermaphrodite gap jn symmetric', header=2, index_col=2).iloc[:469, 2:471]
        self._check(cells)
        self._slice(cells)

    def _check(self, cells):
        """ check if cells exist
        cells: a list of cell names
        """
        chemical_index = set([cell.lower() for cell in self.chemical.index])
        chemical_columns = set([cell.lower() for cell in self.chemical.columns])
        gap_junction_index = set([cell.lower() for cell in self.gap_junction.index])
        gap_junction_columns = set([cell.lower() for cell in self.gap_junction.columns])
        for cell in cells:
            if cell not in self.chemical.index:
                if cell.lower() not in chemical_index:
                    self.chemical.loc[cell, :] = np.full_like(self.chemical.columns, fill_value=np.nan)
                else:
                    raise KeyError('Chemical adjacency rows do not include {}'.format(cell))
            if cell not in self.chemical.columns:
                if cell.lower() not in chemical_columns:
                    self.chemical.loc[:, cell] = np.full_like(self.chemical.index, fill_value=np.nan)
                else:
                    raise KeyError('Chemical adjacency columns do not include {}'.format(cell))
            if cell not in self.gap_junction.index:
                if cell.lower() not in gap_junction_index:
                    self.gap_junction[cell, :] = np.full_like(self.gap_junction.columns, fill_value=np.nan)
                else:
                    raise KeyError('Gap junction adjacency rows do not include {}'.format(cell))
            if cell not in self.gap_junction.columns:
                if cell.lower() not in gap_junction_columns:
                    self.gap_junction[:, cell] = np.full_like(self.gap_junction.index, fill_value=np.nan)
                else:
                    raise KeyError('Gap junction adjacency columns do not include {}'.format(cell))

    def _slice(self, cells):
        # cells: a list of cell names
        self.chemical = self.chemical.loc[cells, cells]
        self.gap_junction = self.chemical.loc[cells, cells]

    def _weight(self):
        chemical = self.chemical.replace(np.nan, 0).to_numpy(dtype=np.int32)
        gap_junction = self.gap_junction.replace(np.nan, 0).to_numpy(dtype=np.int32)
        chemical = torch.from_numpy(chemical)
        gap_junction = torch.from_numpy(gap_junction)
        return chemical, gap_junction

    def _polarity_mask(self, synapses):
        """ chemical synaptic polarity mask
        excitatory mask: if True, the connection is excitatory
        inhibitory mask: if True, the connection is inhibitory
        synapses: [(pre_cells1, post_cells1), (pre_cells2, post_cells2), ...]
        """
        mask = self.chemical.copy()
        mask.loc[:, :] = False
        for pre_cells, post_cells in synapses:
            mask.loc[pre_cells, post_cells] = True
        mask = mask.to_numpy(dtype=np.bool)
        mask = torch.from_numpy(mask)
        return mask

    def mask(self):
        chemical, gap_junction = self._weight()
        mask_c = chemical.bool()  # chemical synapse bool mask
        mask_g = gap_junction.bool()  # gap junction bool mask
        ex_mask_c = self._polarity_mask(self.ex_synapses)
        in_mask_c = self._polarity_mask(self.in_synapses)
        ex_mask_c *= mask_c  # excitatory chemical synapse bool mask
        in_mask_c *= mask_c  # inhibitory chemical synapse bool mask
        if torch.any(ex_mask_c == in_mask_c) is True:
            raise ValueError('There is overlap in excitatory mask and inhibitory mask!')
        return mask_c, mask_g, ex_mask_c, in_mask_c


class SNNCell(torch.nn.Module):
    """ neuronal network model
    https://doi.org/10.1038/s41598-021-92690-2
    """
    def __init__(self, freq, n, p, mask_c, ex_mask_c, in_mask_c, mask_g):
        super(SNNCell, self).__init__()
        self.freq = freq  # freq of data sequence
        self.n = n  # cell count
        self.p = p  # proprioception size
        self.tau = torch.nn.Parameter(torch.rand(n))  # cell time constant, (cell_count, )
        self.w_c = torch.nn.Parameter(torch.rand((n, n)))  # chemical synapse weight, (cell_count, cell_count)
        self.mask_c = torch.nn.Parameter(mask_c, requires_grad=False)  # chemical synapse bool mask, (cell_count, cell_count)
        self.ex_mask_c = torch.nn.Parameter(ex_mask_c, requires_grad=False)  # excitatory chemical synapse bool mask, (cell_count, cell_count)
        self.in_mask_c = torch.nn.Parameter(in_mask_c, requires_grad=False)  # inhibitory chemical synapse bool mask, (cell_count, cell_count)
        self.w_g = torch.nn.Parameter(torch.rand((n, n)))  # gap junction weight, (cell_count, cell_count)
        self.mask_g = torch.nn.Parameter(mask_g, requires_grad=False)  # gap junction bool mask, (cell_count, cell_count)
        self.w_p = torch.nn.Parameter(torch.rand((p, n)))  # proprioception input synapse weight, (proprioception_size, cell_count)
        self.bias = torch.nn.Parameter(torch.rand(n))  # cell state bias, (cell_count, )

    def forward(self, state, activation, proprioception):
        """ forward 1 step
        state: cell state, (batch_size, cell_count)
        activation: cell activation, (batch_size, cell_count)
        proprioception: (batch_size, proprioception_size)
        """
        w_c = self.w_c.abs() * self.ex_mask_c - self.w_c.abs() * self.in_mask_c + self.w_c * (self.mask_c - self.ex_mask_c - self.in_mask_c)
        synapse_input = torch.mm(activation, w_c)  # chemical synapse input, (batch_size, cell_count)
        delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
        w_g = self.w_g.abs() * self.mask_g
        gap_input = torch.sum(delta_state * w_g, dim=1)  # gap junction input, (batch_size, cell_count)
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
