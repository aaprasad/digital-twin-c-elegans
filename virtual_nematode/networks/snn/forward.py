import math
import torch


class SNNCell(torch.nn.Module):
    """ neuronal network model
    https://doi.org/10.1038/s41598-021-92690-2
    """
    def __init__(
        self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index,
        # w_c_ex_mask, w_c_in_mask
    ):
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
        # w_c_mask = w_c_mask ^ (w_c_ex_mask & w_c_mask) ^ (w_c_in_mask & w_c_mask)
        self.w_c_mask = torch.nn.Parameter(w_c_mask, requires_grad=False)  # chemical synapse bool mask, (cell_count, cell_count)
        # self.w_c_ex_mask = torch.nn.Parameter(w_c_ex_mask, requires_grad=False)  # excitatory chemical synapse bool mask, (cell_count, cell_count)
        # self.w_c_in_mask = torch.nn.Parameter(w_c_in_mask, requires_grad=False)  # inhibitory chemical synapse bool mask, (cell_count, cell_count)
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
        # self.w_output = torch.nn.Parameter(torch.zeros(m).uniform_(0, 1))  # muscle activation scaling, (muscle_count, )
        self.output_range = torch.nn.Parameter(torch.sigmoid(torch.tensor([-1, 1])), requires_grad=False)  # hypothetical output range

    @property
    def init_state(self):
        """ initial state and activation """
        state = self.bias.clone().detach()  # (cell_count, )
        activation = self.activation_func(state)  # (cell_count, )
        return state, activation

    def _external_input(self, stimuli):
        """ external input
        stimuli: (batch_size, proprioception_size)
        """
        # proprioception weight
        w_p = self.w_p * self.w_p_mask
        # proprioception input
        external_input = torch.mm(stimuli, w_p) / self.w_p_n  # proprioception input, (batch_size, cell_count)
        return external_input

    def forward(self, state, activation, stimuli):
        """ forward 1 step
        state: cell state, (batch_size, cell_count)
        activation: cell activation, (batch_size, cell_count)
        stimuli: (batch_size, stimuli_size)
        """
        # chemical synapse weight
        # w_c = self.w_c.abs()
        # w_c = w_c * self.w_c_ex_mask - w_c * self.w_c_in_mask + self.w_c * self.w_c_mask
        w_c = self.w_c * self.w_c_mask
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # proprioception input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        dt_tau = self.dt / self.steps / self.tau.clamp(0.01, 0.05)
        for i in range(self.steps):
            synapse_input = torch.mm(activation, w_c) / self.w_c_n  # chemical synapse input, (batch_size, cell_count)
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1) / self.w_g_n  # gap junction input, (batch_size, cell_count)
            total_input = synapse_input + gap_input + external_input  # total input, (batch_size, cell_count)
            # cell state and activation, (batch_size, cell_count)
            state = (1 - dt_tau) * state + dt_tau * total_input
            activation = self.activation_func(state)
            # activation = (activation - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
            # activation = activation.clamp(0, 1)
        # muscle output, (batch_size, muscle_count)
        action = activation[:, self.output_index]
        action = (action - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
        action = action.clamp(0, 1)
        # action = action * self.w_output.clamp(0, 1)
        return state, activation, action


class SigmoidNorm(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super(SigmoidNorm, self).__init__()
        self.min_val = torch.sigmoid(torch.tensor(min_val)).item()
        self.max_val = torch.sigmoid(torch.tensor(max_val)).item()

    def forward(self, input):
        output = (torch.sigmoid(input) - self.min_val) / (self.max_val - self.min_val)
        output = torch.nn.functional.hardtanh(output, 0, 1)
        return output


class SNNCell1(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        super(SNNCell1, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        # self.state_func = torch.nn.Hardtanh(min_val=-3, max_val=3)
        # self.activation_func = torch.nn.Hardsigmoid()
        self.state_func = torch.nn.Tanh()
        self.activation_func = SigmoidNorm(min_val=-1, max_val=1)
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-3, 3))
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.05))
        self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1, 1))
        self.w_c_mask = torch.nn.Parameter(w_c_mask, requires_grad=False)
        self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)
        self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1, 1))
        self.w_p_mask = torch.nn.Parameter(w_p_mask, requires_grad=False)
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)

    @property
    def init_state(self):
        """ initial state and activation """
        # bias = self.bias.clone().detach()
        # state = self.state_func(bias)
        state = torch.zeros(self.n)  # (cell_count, )
        activation = self.activation_func(state)  # (cell_count, )
        return state, activation

    def _external_input(self, stimuli):
        # proprioception input
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p)
        return external_input

    def forward(self, state, activation, stimuli):
        # chemical synapse weight
        w_c = self.w_c * self.w_c_mask
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        dt_tau = self.dt / self.steps / self.tau.clamp(0.01, 0.05)
        for i in range(self.steps):
            # chemical synapse input
            synapse_input = torch.mm(activation, w_c)
            # gap junction input, (batch_size, cell_count)
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1)
            # total input
            total_input = synapse_input + gap_input + external_input
            # cell state and activation
            state = (1 - dt_tau) * state + dt_tau * total_input
            state = self.state_func(state)
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class SNNCell2(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        super(SNNCell2, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.state_func = torch.nn.Tanh()
        self.activation_func = torch.nn.Sigmoid()
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-3, 3))
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.05))
        self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1, 1))
        self.w_c_mask = torch.nn.Parameter(w_c_mask, requires_grad=False)
        self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)
        self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1, 1))
        self.w_p_mask = torch.nn.Parameter(w_p_mask, requires_grad=False)
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)

    @property
    def init_state(self):
        """ initial state and activation """
        value = torch.zeros(self.n)
        state = self.state_func(value)  # (cell_count, )
        activation = self.activation_func(value)  # (cell_count, )
        return state, activation

    def _external_input(self, stimuli):
        # proprioception input
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p)
        return external_input

    def forward(self, state, activation, stimuli):
        # chemical synapse weight
        w_c = self.w_c * self.w_c_mask
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        dt_tau = self.dt / self.steps / self.tau.clamp(0.01, 0.05)
        for i in range(self.steps):
            # chemical synapse input
            synapse_input = torch.mm(activation, w_c)
            # gap junction input, (batch_size, cell_count)
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1)
            # total input
            total_input = synapse_input + gap_input + external_input
            # cell state and activation
            value = (1 - dt_tau) * state + dt_tau * total_input
            state = self.state_func(value)
            activation = self.activation_func(value)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class SNNCell3(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        """ SNN with chemical synapses and gap junctions, including chemical synapse polarity restriction
        w_c_mask[0]: chemical mask with no polarity restriction, already excluded excitatory/inhibitory chemical mask
        w_c_mask[1]: excitatory chemical mask
        w_c_mask[2]: inhibitory chemical mask, no overlap with excitatory chemical mask
        """
        super(SNNCell3, self).__init__()
        self.dt = dt  # env dt
        self.steps = steps  # ode steps
        self.n = n  # cell count
        self.m = m  # muscle count
        self.p = p  # proprioception size
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-3, 3))  # (n, )
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.05))  # (n, )
        self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1, 1))  # (n, n)
        self.w_c_mask = torch.nn.Parameter(w_c_mask, requires_grad=False)  # (3, n, n), bool
        self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))  # (n, n)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1, 1))  # (p, n)
        self.w_p_mask = torch.nn.Parameter(w_p_mask, requires_grad=False)  # (3, p, n), bool
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.state_func = torch.nn.Tanh()
        self.activation_func = torch.nn.Sigmoid()

    @property
    def init_state(self):
        """ initial state and activation """
        value = torch.zeros(self.n)
        state = self.state_func(value)  # (cell_count, )
        activation = self.activation_func(value)  # (cell_count, )
        return state, activation

    def _external_input(self, stimuli):
        # proprioception input
        w_p_abs = self.w_p.abs()
        w_p = self.w_p * self.w_p_mask[0] + w_p_abs * self.w_p_mask[1] - w_p_abs * self.w_p_mask[2]
        external_input = torch.mm(stimuli, w_p)
        return external_input

    def forward(self, state, activation, stimuli):
        # chemical synapse weight
        w_c_abs = self.w_c.abs()
        w_c = self.w_c * self.w_c_mask[0] + w_c_abs * self.w_c_mask[1] - w_c_abs * self.w_c_mask[2]
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        dt_tau = self.dt / self.steps / self.tau.clamp(0.01, 0.05)
        for i in range(self.steps):
            # chemical synapse input
            synapse_input = torch.mm(activation, w_c)
            # gap junction input, (batch_size, cell_count)
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1)
            # total input
            total_input = synapse_input + gap_input + external_input
            # cell state and activation
            value = (1 - dt_tau) * state + dt_tau * total_input
            state = self.state_func(value)
            activation = self.activation_func(value)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class SNNCell4(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        super(SNNCell4, self).__init__()
        self.dt = dt  # env dt
        self.steps = steps  # ode steps
        self.n = n  # cell count
        self.m = m  # muscle count
        self.p = p  # proprioception size
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-3, 3))  # (n, )
        # self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1 / math.sqrt(n), 1 / math.sqrt(n)))
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.05))  # (n, )
        self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1, 1))  # (n, n)
        # self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1 / math.sqrt(n), 1 / math.sqrt(n)))
        self.w_c_mask = torch.nn.Parameter(w_c_mask, requires_grad=False)  # (3, n, n), bool
        self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))  # (n, n)
        # self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1 / math.sqrt(n)))
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1, 1))  # (p, n)
        # self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1 / math.sqrt(p), 1 / math.sqrt(p)))
        self.w_p_mask = torch.nn.Parameter(w_p_mask, requires_grad=False)  # (3, p, n), bool
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        # self.delta_state_func = torch.nn.Tanh()
        self.activation_func = torch.nn.Sigmoid()

    @property
    def init_state(self):
        """ initial state and activation """
        state = self.bias.clone().detach()  # (n, )
        # state = torch.zeros(self.n)
        activation = self.activation_func(state)  # (n, )
        return state, activation

    def _external_input(self, stimuli):
        # proprioception input
        w_p_abs = self.w_p.abs()
        w_p = self.w_p * self.w_p_mask[0] + w_p_abs * self.w_p_mask[1] - w_p_abs * self.w_p_mask[2]
        external_input = torch.mm(stimuli, w_p)
        return external_input

    def forward(self, state, activation, stimuli):
        # chemical synapse weight
        w_c_abs = self.w_c.abs()
        w_c = self.w_c * self.w_c_mask[0] + w_c_abs * self.w_c_mask[1] - w_c_abs * self.w_c_mask[2]
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        dt_tau = self.dt / self.steps / self.tau.clamp(0.01, 0.05)
        for i in range(self.steps):
            # chemical synapse input
            synapse_input = torch.mm(activation, w_c)
            # gap junction input, (batch_size, cell_count)
            # state_tanh = state.tanh()
            # delta_state = state_tanh.unsqueeze(dim=2).repeat(1, 1, self.n) - state_tanh.unsqueeze(dim=1).repeat(1, self.n, 1)
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            # delta_state = self.delta_state_func(delta_state)
            gap_input = torch.sum(delta_state * w_g, dim=1)
            # total input
            total_input = synapse_input + gap_input + external_input
            # cell state and activation
            state = (1 - dt_tau) * state + dt_tau * total_input
            # state = (1 - dt_tau) * state_tanh + dt_tau * total_input
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class SNN(torch.nn.Module):
    def __init__(self, cell):
        super(SNN, self).__init__()
        self.cell = cell

    def init_state(self, batch_size, device):
        """ initial state and activation """
        state, activation = self.cell.init_state  # (cell_count, )
        state = state.unsqueeze(dim=0).repeat(batch_size, 1)
        state = state.to(device=device)
        activation = activation.unsqueeze(dim=0).repeat(batch_size, 1)
        activation = activation.to(device=device)
        return state, activation

    def forward(self, stimuli):
        device = stimuli.device
        batch_size = stimuli.size(0)
        seq_len = stimuli.size(1)
        # initial state and activation
        state, activation = self.init_state(batch_size, device)
        # simulate sequence
        actions = []
        for t in range(seq_len):
            s = stimuli[:, t]
            state, activation, action = self.cell.forward(state, activation, s)
            actions.append(action)  # action, (batch_size, muscle_count)
        actions = torch.stack(actions, dim=1)  # action sequence, (batch_size, seq_len, muscle_count)
        return actions

    def step(self, state, activation, stimuli):
        if state is None or activation is None:
            device = stimuli.device
            batch_size = stimuli.size(0)
            state, activation = self.init_state(batch_size, device)
        state, activation, action = self.cell.forward(state, activation, stimuli)
        return state, activation, action
