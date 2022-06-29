import numpy as np
import pandas as pd
import torch


class DummyConnectome(object):
    """ fully connected chemical connectome
    * fully connected chemical connections, no polarity constraints
    * no gap junction connection
    * fully connected proprioception input to all cells
    """
    def __init__(self, neurons, muscles, p, p_mask):
        self.neurons = neurons
        self.muscles = muscles
        self.cells = neurons + muscles
        self.p = p
        self.p_mask = p_mask
        self.chemical, self.gap_junction = self._init()

    def _init(self):
        chemical = pd.DataFrame(True, index=self.cells, columns=self.cells)
        gap_junction = pd.DataFrame(False, index=self.cells, columns=self.cells)
        return chemical, gap_junction

    def _proprioception_mask(self):
        """ proprioception input synapse bool mask
        https://doi.org/10.1016/j.neuron.2012.08.039
        """
        mask = pd.DataFrame(False, index=list(range(self.p)), columns=self.cells)
        mask.loc[:, self.neurons] = True
        mask = torch.from_numpy(mask.to_numpy(dtype=np.bool))
        if self.p_mask is False:  # no proprioception mask: all cells receive proprioception input
            mask = torch.full_like(mask, fill_value=True)
        return mask

    def _weight(self):
        w_c_mask = torch.from_numpy(self.chemical.to_numpy(dtype=np.bool))
        w_g_mask = torch.from_numpy(self.gap_junction.to_numpy(dtype=np.bool))
        return w_c_mask, w_g_mask

    def _polarity_mask(self, **kwargs):
        mask = pd.DataFrame(False, index=self.cells, columns=self.cells)
        mask = torch.from_numpy(mask.to_numpy(dtype=np.bool))
        return mask

    def _polarity_masks(self):
        w_c_ex_mask = self._polarity_mask()
        w_c_in_mask = self._polarity_mask()
        return w_c_ex_mask, w_c_in_mask

    def mask(self):
        """ mask generatation
        chemical mask
            mask_ij is True -> w_ij
            mask_ij is False -> w_ij = 0
        gap junction mask: symmetric
            mask_ij is True -> g_ij, g_ji
            mask_ij is False -> g_ij = g_ji = 0
        excitatory chemical mask: a sub mask of chemical mask
            mask_ij is True -> w_ij >= 0
            mask_ij is False -> w_ij
        inhibitory chemical mask: a sub mask of chemical mask, no overlap with excitatory chemical mask
            mask_ij is True -> w_ij <= 0
            mask_ij is False -> w_ij
        proprioception mask
            mask_pi is True -> w_pi
        """
        w_c_mask, w_g_mask = self._weight()
        if torch.all(w_g_mask.tril() == w_g_mask.triu().T).item() is not True:
            raise AssertionError('Gap junction mask is not symmetric!')
        w_c_ex_mask, w_c_in_mask = self._polarity_masks()
        w_c_ex_mask = w_c_ex_mask & w_c_mask
        w_c_in_mask = w_c_in_mask & w_c_mask
        if torch.any(w_c_ex_mask & w_c_in_mask).item() is True:
            raise AssertionError('There is overlap in excitatory mask and inhibitory mask!')
        muscles = set(self.muscles)
        w_p_mask = self._proprioception_mask()
        output_index = torch.tensor([True if cell in muscles else False for cell in self.cells])
        return w_c_mask, w_g_mask, w_c_ex_mask, w_c_in_mask, w_p_mask, output_index


class Connectome(DummyConnectome):
    def __init__(self, neurons, muscles, ex_synapses, in_synapses, path, p, p_mask, polarity_mask):
        self.ex_synapses = ex_synapses
        self.in_synapses = in_synapses
        self.path = path
        self.polarity_mask = polarity_mask
        super(Connectome, self).__init__(neurons, muscles, p, p_mask)

    def _init(self):
        chemical = pd.read_excel(self.path, sheet_name='hermaphrodite chemical', header=2, index_col=2).iloc[:300, 2:456]
        gap_junction = pd.read_excel(self.path, sheet_name='hermaphrodite gap jn symmetric', header=2, index_col=2).iloc[:469, 2:471]
        self._check(gap_junction)
        chemical = self._add(chemical)
        gap_junction = self._add(gap_junction)
        chemical = self._slice(chemical)
        gap_junction = self._slice(gap_junction)
        return chemical, gap_junction

    def _check(self, gap_junction):
        """ check if cells exist
        cells: a list of cell names
        """
        all_cells = set(list(gap_junction.index))
        for cell in self.cells:
            if cell not in all_cells:
                raise AssertionError('Cell {} does not exist!'.format(cell))

    def _add(self, adjacency):
        for cell in self.cells:
            if cell not in adjacency.index:
                adjacency.loc[cell, :] = np.full_like(adjacency.columns, fill_value=np.nan)
            if cell not in adjacency.columns:
                adjacency.loc[:, cell] = np.full_like(adjacency.index, fill_value=np.nan)
        return adjacency

    def _slice(self, adjacency):
        adjacency = adjacency.loc[self.cells, self.cells]
        return adjacency

    def _weight(self):
        chemical = self.chemical.replace(np.nan, 0)
        gap_junction = self.gap_junction.replace(np.nan, 0)
        w_c_mask = torch.from_numpy(chemical.to_numpy(dtype=np.bool))
        w_g_mask = torch.from_numpy(gap_junction.to_numpy(dtype=np.bool))
        return w_c_mask, w_g_mask

    def _polarity_mask(self, synapses):
        """ chemical synaptic polarity mask
        excitatory mask: if True, the connection is excitatory
        inhibitory mask: if True, the connection is inhibitory
        synapses: [(pre_cells1, post_cells1), (pre_cells2, post_cells2), ...]
        """
        mask = pd.DataFrame(False, index=self.cells, columns=self.cells)
        if self.polarity_mask is True:
            for pre_cells, post_cells in synapses:
                mask.loc[pre_cells, post_cells] = True
        mask = torch.from_numpy(mask.to_numpy(dtype=np.bool))
        return mask

    def _polarity_masks(self):
        w_c_ex_mask = self._polarity_mask(self.ex_synapses)
        w_c_in_mask = self._polarity_mask(self.in_synapses)
        return w_c_ex_mask, w_c_in_mask


class SNNCell(torch.nn.Module):
    """ neuronal network model
    https://doi.org/10.1038/s41598-021-92690-2
    """
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_c_ex_mask, w_c_in_mask, w_g_mask, w_p_mask, output_index):
        super(SNNCell, self).__init__()
        self.dt = dt  # env dt
        self.steps = steps  # ode steps
        self.n = n  # cell count
        self.m = m  # muscle count
        self.p = p  # proprioception size
        self.activation_func = torch.nn.Sigmoid()
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # cell state bias, (cell_count, )
        # self.tau = torch.nn.Parameter(torch.zeros(n).normal_(mean=0.08, std=0.01))
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.05))  # cell time constant, (cell_count, )
        self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1, 1))  # chemical synapse weight, (cell_count, cell_count)
        # exclude excitatory/inhibitory synapse
        w_c_mask = w_c_mask ^ (w_c_ex_mask & w_c_mask) ^ (w_c_in_mask & w_c_mask)
        self.w_c_mask = torch.nn.Parameter(w_c_mask, requires_grad=False)  # chemical synapse bool mask, (cell_count, cell_count)
        self.w_c_ex_mask = torch.nn.Parameter(w_c_ex_mask, requires_grad=False)  # excitatory chemical synapse bool mask, (cell_count, cell_count)
        self.w_c_in_mask = torch.nn.Parameter(w_c_in_mask, requires_grad=False)  # inhibitory chemical synapse bool mask, (cell_count, cell_count)
        w_c_n = w_c_mask.sum(dim=0)
        w_c_n[w_c_n == 0] = 1
        self.w_c_n = torch.nn.Parameter(w_c_n, requires_grad=False)  # input chemical synapse amount, (cell_count, )
        self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))  # gap junction weight, (cell_count, cell_count)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # gap junction bool mask, (cell_count, cell_count)
        w_g_n = w_g_mask.sum(dim=0)
        w_g_n[w_g_n == 0] = 1
        self.w_g_n = torch.nn.Parameter(w_g_n, requires_grad=False)  # input gap junction amount, (cell_count, )
        self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1, 1))  # proprioception input synapse weight, (proprioception_size, cell_count)
        self.w_p_mask = torch.nn.Parameter(w_p_mask, requires_grad=False)  # proprioception input synapse bool mask, (proprioception_size, cell_count)
        w_p_n = w_p_mask.sum(dim=0)
        w_p_n[w_p_n == 0] = 1
        self.w_p_n = torch.nn.Parameter(w_p_n, requires_grad=False)  # proprioception input synapse amount, (cell_count, )
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # muscle output mask, (cell_count, )
        self.w_output = torch.nn.Parameter(torch.zeros(m).uniform_(0, 1))  # muscle activation scaling, (muscle_count, )

    @property
    def init_state(self):
        return self.bias.clone().detach()  # (cell_count, )

    def forward(self, state, activation, proprioception):
        """ forward 1 step
        state: cell state, (batch_size, cell_count)
        activation: cell activation, (batch_size, cell_count)
        proprioception: (batch_size, proprioception_size)
        """
        # chemical synapse weight
        w_c = self.w_c.abs()
        w_c = w_c * self.w_c_ex_mask - w_c * self.w_c_in_mask + self.w_c * self.w_c_mask
        # gap junction weight
        w_g = self.w_g.abs().clamp(0, 1)
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # proprioception weight
        w_p = self.w_p * self.w_p_mask
        # proprioception input
        external_input = torch.mm(proprioception, w_p) / self.w_p_n  # proprioception input, (batch_size, cell_count)
        # proprioception input + bias
        external_input += self.bias
        # dt / tau
        dt_tau = self.dt / self.steps / self.tau.clamp(0.01, 0.05)
        for i in range(self.steps):
            synapse_input = torch.mm(activation, w_c) / self.w_c_n  # chemical synapse input, (batch_size, cell_count)
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1) / self.w_g_n  # gap junction input, (batch_size, cell_count)
            total_input = synapse_input + gap_input + external_input  # total input, (batch_size, cell_count)
            # cell state, (batch_size, cell_count)
            state = (1 - dt_tau) * state + dt_tau * total_input
            # cell activation, (batch_size, cell_count)
            activation = self.activation_func(state)
        # muscle output, (batch_size, muscle_count)
        action = activation[:, self.output_index] * self.w_output.clamp(0, 1)
        return state, activation, action


class SNN(torch.nn.Module):
    def __init__(self, cell):
        super(SNN, self).__init__()
        self.cell = cell

    def forward(self, proprioceptions):
        device = proprioceptions.device
        batch_size = proprioceptions.size(0)
        seq_len = proprioceptions.size(1)
        # initial state and activation
        state = self.cell.init_state  # (cell_count, )
        state = state.unsqueeze(dim=0).repeat(batch_size, 1)
        state = state.to(device=device)
        activation = self.cell.activation_func(state)
        # simulate sequence
        actions = []
        for t in range(seq_len):
            p = proprioceptions[:, t]
            state, activation, action = self.cell.forward(state, activation, p)
            actions.append(action)  # action, (batch_size, muscle_count)
        actions = torch.stack(actions, dim=1)  # action sequence, (batch_size, seq_len, muscle_count)
        return actions

    def step(self, state, activation, proprioception):
        if state is None or activation is None:
            device = proprioception.device
            batch_size = proprioception.size(0)
            state = self.cell.init_state  # (cell_count, )
            state = state.unsqueeze(dim=0).repeat(batch_size, 1)
            state = state.to(device=device)
            activation = self.cell.activation_func(state)
        state, activation, action = self.cell.forward(state, activation, proprioception)
        return state, activation, action
