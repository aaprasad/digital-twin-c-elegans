import torch


class SNNCell(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        super(SNNCell, self).__init__()
        self.dt = dt  # env dt
        self.steps = steps  # ode steps
        self.n = n  # number of cells
        self.m = m  # number of muscles
        self.p = p  # proprioception input size
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.05))  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # (n, )
        self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1, 1))  # (n, n)
        self.w_c_mask = torch.nn.Parameter(w_c_mask, requires_grad=False)  # (3, n, n), bool
        self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))  # (n, n)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1, 1))  # (p, n)
        self.w_p_mask = torch.nn.Parameter(w_p_mask, requires_grad=False)  # (3, p, n), bool
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.activation_func = torch.nn.Sigmoid()

    @property
    def init_state(self):
        """ initial state and activation """
        state = self.bias.clone().detach()  # (n, )
        activation = self.activation_func(state)  # (n, )
        return state, activation

    def _external_input(self, stimuli):
        """ proprioception input """
        w_p_abs = self.w_p.abs()
        w_p = self.w_p * self.w_p_mask[0] + w_p_abs * self.w_p_mask[1] - w_p_abs * self.w_p_mask[2]
        external_input = torch.mm(stimuli, w_p)  # (n, )
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
            # gap junction input
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1)
            # total input
            total_input = synapse_input + gap_input + external_input
            # cell state and activation
            state = (1 - dt_tau) * state + dt_tau * total_input
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class SNNCell1(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        super(SNNCell1, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.05))  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # (n, )
        self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1, 1))  # (n, n)
        self.w_c_mask = torch.nn.Parameter(w_c_mask, requires_grad=False)  # (3, n, n), bool
        w_c_n = w_c_mask.sum(dim=[0, 1])
        w_c_n[w_c_n == 0] = 1
        self.w_c_n = torch.nn.Parameter(w_c_n, requires_grad=False)  # (n, )
        self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))  # (n, n)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        w_g_n = w_g_mask.sum(dim=0)
        w_g_n[w_g_n == 0] = 1
        self.w_g_n = torch.nn.Parameter(w_g_n, requires_grad=False)  # (n, )
        self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1, 1))  # (p, n)
        self.w_p_mask = torch.nn.Parameter(w_p_mask, requires_grad=False)  # (3, p, n), bool
        w_p_n = w_p_mask.sum(dim=[0, 1])
        w_p_n[w_p_n == 0] = 1
        self.w_p_n = torch.nn.Parameter(w_p_n, requires_grad=False)  # (n, )
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.activation_func = torch.nn.Sigmoid()
        self.action_scaling = torch.sigmoid(torch.tensor([-1, 1])).tolist()

    @property
    def init_state(self):
        """ initial state and activation """
        state = self.bias.clone().detach()  # (n, )
        activation = self.activation_func(state)  # (n, )
        return state, activation

    def _external_input(self, stimuli):
        """ proprioception input """
        w_p_abs = self.w_p.abs()
        w_p = self.w_p * self.w_p_mask[0] + w_p_abs * self.w_p_mask[1] - w_p_abs * self.w_p_mask[2]
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
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
            synapse_input = torch.mm(activation, w_c) / self.w_c_n
            # gap junction input
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1) / self.w_g_n
            # total input
            total_input = synapse_input + gap_input + external_input
            # cell state and activation
            state = (1 - dt_tau) * state + dt_tau * total_input
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        action = (action - self.action_scaling[0]) / (self.action_scaling[1] - self.action_scaling[0])
        action = action.clamp(0, 1)
        return state, activation, action


class SNNCell3(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        super(SNNCell3, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.05))  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # (n, )
        self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1, 1))  # (n, n)
        self.w_c_mask = torch.nn.Parameter(w_c_mask, requires_grad=False)  # (3, n, n), bool
        w_c_n = w_c_mask.sum(dim=[0, 1])
        w_c_n[w_c_n == 0] = 1
        self.w_c_n = torch.nn.Parameter(w_c_n, requires_grad=False)  # (n, )
        self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))  # (n, n)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        w_g_n = w_g_mask.sum(dim=0)
        w_g_n[w_g_n == 0] = 1
        self.w_g_n = torch.nn.Parameter(w_g_n, requires_grad=False)  # (n, )
        self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1, 1))  # (p, n)
        self.w_p_mask = torch.nn.Parameter(w_p_mask, requires_grad=False)  # (3, p, n), bool
        w_p_n = w_p_mask.sum(dim=[0, 1])
        w_p_n[w_p_n == 0] = 1
        self.w_p_n = torch.nn.Parameter(w_p_n, requires_grad=False)  # (n, )
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.state_func = torch.nn.Hardtanh(-1, 1)
        self.activation_func = torch.nn.Sigmoid()
        self.activation_scaling = torch.sigmoid(torch.tensor([-1, 1])).tolist()

    @property
    def init_state(self):
        """ initial state and activation """
        state = self.bias.clone().detach()
        state = self.state_func(state)
        activation = self.activation_func(state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        return state, activation  # (n, ), (n, )

    def _external_input(self, stimuli):
        """ proprioception input """
        w_p_abs = self.w_p.abs()
        w_p = self.w_p * self.w_p_mask[0] + w_p_abs * self.w_p_mask[1] - w_p_abs * self.w_p_mask[2]
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
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
            synapse_input = torch.mm(activation, w_c) / self.w_c_n
            # gap junction input
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1) / self.w_g_n
            # total input
            total_input = synapse_input + gap_input + external_input
            # cell state and activation
            state = (1 - dt_tau) * state + dt_tau * total_input
            state = self.state_func(state)
            activation = self.activation_func(state)
            activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
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
