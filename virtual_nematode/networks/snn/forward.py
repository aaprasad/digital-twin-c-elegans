import numpy as np
import pandas as pd
import torch


class Connectome(object):
    def __init__(self, neurons, muscles, ex_synapses, in_synapses, path):
        self.neurons = neurons
        self.muscles = muscles
        self.cells = neurons + muscles
        self.ex_synapses = ex_synapses
        self.in_synapses = in_synapses
        self.chemical = pd.read_excel(path, sheet_name='hermaphrodite chemical', header=2, index_col=2).iloc[:300, 2:456]
        self.gap_junction = pd.read_excel(path, sheet_name='hermaphrodite gap jn symmetric', header=2, index_col=2).iloc[:469, 2:471]
        self._check(self.cells)
        self._slice(self.cells)

    def _check(self, cells):
        """ check if cells exist
        cells: a list of cell names
        """
        all_cells = set(
            list(self.chemical.index) + list(self.chemical.columns) +
            list(self.gap_junction.index) + list(self.gap_junction.columns)
        )
        for cell in cells:
            if cell not in all_cells:
                raise AssertionError('Cell {} does not exist!'.format(cell))
        for cell in cells:
            if cell not in self.chemical.index:
                self.chemical.loc[cell, :] = np.full_like(self.chemical.columns, fill_value=np.nan)
            if cell not in self.chemical.columns:
                self.chemical.loc[:, cell] = np.full_like(self.chemical.index, fill_value=np.nan)
            if cell not in self.gap_junction.index:
                self.gap_junction.loc[cell, :] = np.full_like(self.gap_junction.columns, fill_value=np.nan)
            if cell not in self.gap_junction.columns:
                self.gap_junction.loc[:, cell] = np.full_like(self.gap_junction.index, fill_value=np.nan)

    def _slice(self, cells):
        # cells: a list of cell names
        self.chemical = self.chemical.loc[cells, cells]
        self.gap_junction = self.gap_junction.loc[cells, cells]

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
        mask = pd.DataFrame(False, index=self.cells, columns=self.cells)
        for pre_cells, post_cells in synapses:
            mask.loc[pre_cells, post_cells] = True
        mask = mask.to_numpy(dtype=np.bool)
        mask = torch.from_numpy(mask)
        return mask

    def proprioception_mask(self, p, p_mask=True):
        """ proprioception input synapse bool mask
        https://doi.org/10.1016/j.neuron.2012.08.039
        """
        mask = pd.DataFrame(False, index=list(range(p)), columns=self.cells)
        mask.loc[:, self.neurons] = True
        mask = mask.to_numpy(dtype=np.bool)
        mask = torch.from_numpy(mask)
        if p_mask is False:  # no proprioception mask: all cells receive proprioception input
            mask = torch.full_like(mask, fill_value=True)
        return mask

    def mask(self, polarity_mask=True):
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
        """
        chemical, gap_junction = self._weight()
        mask_c = chemical.bool()  # chemical synapse bool mask
        mask_g = gap_junction.bool()  # gap junction bool mask
        ex_mask_c = self._polarity_mask(self.ex_synapses)
        in_mask_c = self._polarity_mask(self.in_synapses)
        if polarity_mask is False:  # no chemical synapse polarity mask: no polarity restraint
            ex_mask_c = torch.full_like(ex_mask_c, fill_value=False)
            in_mask_c = torch.full_like(in_mask_c, fill_value=False)
        ex_mask_c *= mask_c  # excitatory chemical synapse bool mask
        in_mask_c *= mask_c  # inhibitory chemical synapse bool mask
        if torch.any(ex_mask_c & in_mask_c) is True:
            raise AssertionError('There is overlap in excitatory mask and inhibitory mask!')
        muscles = set(self.muscles)
        mask_output = torch.tensor([True if cell in muscles else False for cell in self.cells])
        return mask_c, mask_g, ex_mask_c, in_mask_c, mask_output


class LinearConnectome(object):
    """ fully connected chemical connectome
    * fully connected chemical connections, no polarity constraints
    * no gap junction connection
    * fully connected proprioception input to all cells
    """
    def __init__(self, neurons, muscles):
        self.neurons = neurons
        self.muscles = muscles
        self.cells = neurons + muscles
        self.chemical = pd.DataFrame(True, index=self.cells, columns=self.cells)
        self.gap_junction = pd.DataFrame(False, index=self.cells, columns=self.cells)

    def proprioception_mask(self, p, p_mask=True):
        mask = pd.DataFrame(False, index=list(range(p)), columns=self.cells)
        mask.loc[:, self.neurons] = True
        mask = mask.to_numpy(dtype=np.bool)
        mask = torch.from_numpy(mask)
        if p_mask is False:  # no proprioception mask: all cells receive proprioception input
            mask = torch.full_like(mask, fill_value=True)
        return mask

    def _polarity_mask(self):
        mask = pd.DataFrame(False, index=self.cells, columns=self.cells)
        mask = mask.to_numpy(dtype=np.bool)
        mask = torch.from_numpy(mask)
        return mask

    def mask(self, **kwargs):
        mask_c = torch.from_numpy(self.chemical.to_numpy(dtype=np.bool))
        mask_g = torch.from_numpy(self.gap_junction.to_numpy(dtype=np.bool))
        ex_mask_c = self._polarity_mask()
        in_mask_c = self._polarity_mask()
        ex_mask_c *= mask_c
        in_mask_c *= mask_c
        if torch.any(ex_mask_c & in_mask_c) is True:
            raise AssertionError('There is overlap in excitatory mask and inhibitory mask!')
        muscles = set(self.muscles)
        mask_output = torch.tensor([True if cell in muscles else False for cell in self.cells])
        return mask_c, mask_g, ex_mask_c, in_mask_c, mask_output


class SNNCell(torch.nn.Module):
    """ neuronal network model
    https://doi.org/10.1038/s41598-021-92690-2
    """
    def __init__(self, dt, steps, n, m, p, activation_type, mask_c, ex_mask_c, in_mask_c, mask_g, mask_p, mask_output):
        super(SNNCell, self).__init__()
        self.dt = dt  # env dt
        self.steps = steps  # ode steps
        self.n = n  # cell count
        self.m = m  # muscle count
        self.p = p  # proprioception size
        self.activation_type = activation_type
        if activation_type == 'sigmoid':
            self.activation_func = torch.nn.Sigmoid()
            self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # cell state bias, (cell_count, )
        elif activation_type == 'tanh':
            self.activation_func = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.ReLU()
            )
            self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(0, 1))  # cell state bias, (cell_count, )
        else:
            raise ValueError('Invalid activation func type {}'.format(activation_type))
        # self.tau = torch.nn.Parameter(torch.zeros(n).normal_(mean=0.08, std=0.01))
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.05))  # cell time constant, (cell_count, )
        self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1, 1))  # chemical synapse weight, (cell_count, cell_count)
        self.mask_c = torch.nn.Parameter(mask_c, requires_grad=False)  # chemical synapse bool mask, (cell_count, cell_count)
        self.ex_mask_c = torch.nn.Parameter(ex_mask_c, requires_grad=False)  # excitatory chemical synapse bool mask, (cell_count, cell_count)
        self.in_mask_c = torch.nn.Parameter(in_mask_c, requires_grad=False)  # inhibitory chemical synapse bool mask, (cell_count, cell_count)
        self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))  # gap junction weight, (cell_count, cell_count)
        self.mask_g = torch.nn.Parameter(mask_g, requires_grad=False)  # gap junction bool mask, (cell_count, cell_count)
        self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1, 1))  # proprioception input synapse weight, (proprioception_size, cell_count)
        self.mask_p = torch.nn.Parameter(mask_p, requires_grad=False)  # proprioception input synapse bool mask, (proprioception_size, cell_count)
        self.mask_output = torch.nn.Parameter(mask_output, requires_grad=False)  # muscle output mask, (cell_count, )
        self.w_output = torch.nn.Parameter(torch.zeros(m).uniform_(0, 1))  # muscle activation scaling, (muscle_count, )

    @property
    def state_size(self):
        return self.n

    def forward(self, state, activation, proprioception):
        """ forward 1 step
        state: cell state, (batch_size, cell_count)
        activation: cell activation, (batch_size, cell_count)
        proprioception: (batch_size, proprioception_size)
        """
        # chemical synapse weight
        w_c = self.w_c.abs() * self.ex_mask_c - self.w_c.abs() * self.in_mask_c + self.w_c * (self.mask_c.float() - self.ex_mask_c.float() - self.in_mask_c.float())
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.mask_g
        # proprioception weight
        w_p = self.w_p * self.mask_p
        proprioception_input = torch.mm(proprioception, w_p)  # proprioception input, (batch_size, cell_count)
        # time constant
        dt = self.dt / self.steps
        # tau = self.tau.clamp(min=dt)
        tau = self.tau.clamp(0.01, 0.05)
        for i in range(self.steps):
            synapse_input = torch.mm(activation, w_c)  # chemical synapse input, (batch_size, cell_count)
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1)  # gap junction input, (batch_size, cell_count)
            if self.activation_type == 'tanh':
                bias = self.bias.abs()
            else:
                bias = self.bias
            total_input = synapse_input + gap_input + proprioception_input + bias  # total input, (batch_size, cell_count)
            # cell state, (batch_size, cell_count)
            state = (1 - dt / tau) * state + dt / tau * total_input
            # cell activation, (batch_size, cell_count)
            activation = self.activation_func(state)
        # muscle output, (batch_size, muscle_count)
        action = activation[:, self.mask_output] * self.w_output.clamp(0, 1)
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
        state = self.cell.bias.clone().detach()  # (cell_count, )
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
            state = self.cell.bias.clone().detach()  # (cell_count, )
            state = state.unsqueeze(dim=0).repeat(batch_size, 1)
            state = state.to(device=device)
            activation = self.cell.activation_func(state)
        state, activation, action = self.cell.forward(state, activation, proprioception)
        return state, activation, action
