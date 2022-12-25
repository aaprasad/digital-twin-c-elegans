""" LIF neuron model """

import torch


class RateCell(torch.nn.Module):
    """ rate-coding LIF neuron model
    tau * dv/dt = -v + r * I
    I = x * W_I^T  + v * W_R^T
    reference: Long short-term memory and learning-to-learn in networks of spiking neurons
    """
    def __init__(self, tau, r):
        super(RateCell, self).__init__()
        self.tau = tau
        self.r = r  # resistance

    def forward(self, inputs, states):
        """ params
        inputs: I, external input
        states: v, membrane potential
        """
        if self.feedback is False:
            inputs_prime = self.linear(inputs)
        for i in range(self.unfolds):
            if self.feedback is True:
                inputs_prime = self.linear(torch.cat((inputs, states), dim=-1))  # concat input and new hidden
            states_prime = (-states + inputs_prime) / self.tau  # dy/dt
            states = states + self.delta_t * states_prime
            if self.cell_clip > 0:
                states = torch.clamp(states, min=-self.cell_clip, max=self.cell_clip)
        outputs = self.readout(states)
        return outputs, states


class LeakyIntegratorCurrentBased(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        super(LeakyIntegratorCurrentBased, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        # bias = torch.zeros(n).normal_(mean=0, std=0.5)
        # bias = -1 + bias.abs()
        # self.bias = torch.nn.Parameter(bias)  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # (n, )
        """ chemical synapse init with dedicated excitatory/inhibitory ratio """
        w_c = torch.zeros((n, n)).uniform_(-1, 1)
        polarity = torch.zeros((n, n)).bernoulli_(p=(w_c_mask.sum() * 0.8 - w_c_mask[1].sum()) / w_c_mask[0].sum())
        polarity[polarity == 0.] = -1.
        w_c = w_c.abs() * polarity
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        # self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(-1, 1))  # (n, n)
        """"""
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
        self.w_p_mask = torch.nn.Parameter(w_p_mask.any(dim=0), requires_grad=False)  # (p, n), bool
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
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        return state, activation  # (n, ), (n, )

    def _external_input(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
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
        dt_tau = self.dt / self.steps / self.tau.clamp(0.01, 0.2)
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


class LeakyIntegratorConductanceBased(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        super(LeakyIntegratorConductanceBased, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        # bias = torch.zeros(n).normal_(mean=0, std=0.5)
        # bias = -1 + bias.abs()
        # self.bias = torch.nn.Parameter(bias)  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.05)
        e_c = (0 + e_c) * w_c_mask[0] + (1 + e_c) * w_c_mask[1] + (-1 + e_c) * w_c_mask[2]
        # e_c = (0 + e_c) * w_c_mask[0] + (1 - e_c.abs()) * w_c_mask[1] + (-1 + e_c.abs()) * w_c_mask[2]
        self.e_c = torch.nn.Parameter(e_c)  # (n, n)
        self.w_c_mask = torch.nn.Parameter(w_c_mask.any(dim=0), requires_grad=False)  # (n, n), bool
        w_c_n = w_c_mask.sum(dim=[0, 1])
        w_c_n[w_c_n == 0] = 1
        self.w_c_n = torch.nn.Parameter(w_c_n, requires_grad=False)  # (n, )
        w_g = torch.zeros((n, n)).uniform_(0, 1)
        w_g *= w_g_mask
        self.w_g = torch.nn.Parameter(w_g)  # (n, n)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        w_g_n = w_g_mask.sum(dim=0)
        w_g_n[w_g_n == 0] = 1
        self.w_g_n = torch.nn.Parameter(w_g_n, requires_grad=False)  # (n, )
        w_p = torch.zeros((p, n)).uniform_(-1, 1)
        w_p *= w_p_mask.any(dim=0)
        self.w_p = torch.nn.Parameter(w_p)  # (p, n)
        self.w_p_mask = torch.nn.Parameter(w_p_mask.any(dim=0), requires_grad=False)  # (p, n), bool
        w_p_n = w_p_mask.sum(dim=[0, 1])
        w_p_n[w_p_n == 0] = 1
        self.w_p_n = torch.nn.Parameter(w_p_n, requires_grad=False)  # (n, )
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.state_func = torch.nn.Hardtanh(-1, 1)
        self.activation_func = torch.nn.Sigmoid()
        self.activation_scaling = torch.sigmoid(torch.tensor([-1, 1])).tolist()
        # self.alpha_m = torch.nn.Parameter(torch.ones(m))  # (m, )
        # self.beta_m = torch.nn.Parameter(torch.zeros(m))  # (m, )

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        return state, activation  # (n, ), (n, )

    def _external_input(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
        return external_input

    def forward(self, state, activation, stimuli):
        # chemical synapse weight
        w_c = self.w_c.abs() * self.w_c_mask
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        dt_tau = self.dt / self.steps / self.tau.clamp(0.01, 0.2)
        for i in range(self.steps):
            # chemical synapse input
            synapse_input = (torch.mm(activation, w_c * self.e_c) - torch.mm(activation, w_c) * state) / self.w_c_n
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
        # action = action * self.alpha_m.clamp(0, 1) + self.beta_m.clamp(0, 1)
        return state, activation, action


class LeakyIntegratorConductanceBasedPolarity(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        super(LeakyIntegratorConductanceBasedPolarity, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        # bias = torch.zeros(n).normal_(mean=0, std=0.5)
        # bias = -1 + bias.abs()
        # self.bias = torch.nn.Parameter(bias)  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        self.w_c_mask = torch.nn.Parameter(w_c_mask.any(dim=0), requires_grad=False)  # (n, n), bool
        # e_c = torch.zeros((n, n)).uniform_(-1, 1)
        # e_c = e_c * w_c_mask[0] + 1 * w_c_mask[1] - 1 * w_c_mask[2]
        e_c = torch.zeros((n, n)).uniform_(-3, 3)
        e_c = e_c * w_c_mask[0] + 3. * w_c_mask[1] - 3. * w_c_mask[2]
        self.e_c = torch.nn.Parameter(e_c)  # (n, n)
        w_c_n = w_c_mask.sum(dim=[0, 1])
        w_c_n[w_c_n == 0] = 1
        self.w_c_n = torch.nn.Parameter(w_c_n, requires_grad=False)  # (n, )
        w_g = torch.zeros((n, n)).uniform_(0, 1)
        w_g *= w_g_mask
        self.w_g = torch.nn.Parameter(w_g)  # (n, n)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        w_g_n = w_g_mask.sum(dim=0)
        w_g_n[w_g_n == 0] = 1
        self.w_g_n = torch.nn.Parameter(w_g_n, requires_grad=False)  # (n, )
        w_p = torch.zeros((p, n)).uniform_(-1, 1)
        w_p *= w_p_mask.any(dim=0)
        self.w_p = torch.nn.Parameter(w_p)  # (p, n)
        self.w_p_mask = torch.nn.Parameter(w_p_mask.any(dim=0), requires_grad=False)  # (p, n), bool
        w_p_n = w_p_mask.sum(dim=[0, 1])
        w_p_n[w_p_n == 0] = 1
        self.w_p_n = torch.nn.Parameter(w_p_n, requires_grad=False)  # (n, )
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.state_func = torch.nn.Hardtanh(-1, 1)
        self.activation_func = torch.nn.Sigmoid()
        self.activation_scaling = torch.sigmoid(torch.tensor([-1, 1])).tolist()
        # self.alpha_m = torch.nn.Parameter(torch.ones(m))  # (m, )
        # self.beta_m = torch.nn.Parameter(torch.zeros(m))  # (m, )

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        return state, activation  # (n, ), (n, )

    def _external_input(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
        return external_input

    def forward(self, state, activation, stimuli):
        # chemical synapse weight
        w_c = self.w_c.abs() * self.w_c_mask
        # e_c = self.e_c.clamp(-1, 1)
        e_c = self.e_c.tanh()
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        dt_tau = self.dt / self.steps / self.tau.clamp(0.01, 0.2)
        for i in range(self.steps):
            # chemical synapse input
            synapse_input = (torch.mm(activation, w_c * e_c) - torch.mm(activation, w_c) * state) / self.w_c_n
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
        # action = action * self.alpha_m.clamp(0, 1) + self.beta_m.clamp(0, 1)
        return state, activation, action


class LeakyIntegratorConductanceBasedCalciumFluorescence(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index):
        super(LeakyIntegratorConductanceBasedCalciumFluorescence, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        # bias = torch.zeros(n).normal_(mean=0, std=0.5)
        # bias = -1 + bias.abs()
        # self.bias = torch.nn.Parameter(bias)  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # (n, )
        self.w_c = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.05)
        e_c = (0 + e_c) * w_c_mask[0] + (1 + e_c) * w_c_mask[1] + (-1 + e_c) * w_c_mask[2]
        # e_c = (0 + e_c) * w_c_mask[0] + (1 - e_c.abs()) * w_c_mask[1] + (-1 + e_c.abs()) * w_c_mask[2]
        self.e_c = torch.nn.Parameter(e_c)  # (n, n)
        self.w_c_mask = torch.nn.Parameter(w_c_mask.any(dim=0), requires_grad=False)  # (n, n), bool
        w_c_n = w_c_mask.sum(dim=[0, 1])
        w_c_n[w_c_n == 0] = 1
        self.w_c_n = torch.nn.Parameter(w_c_n, requires_grad=False)  # (n, )
        self.w_g = torch.nn.Parameter(torch.zeros((n, n)).uniform_(0, 1))  # (n, n)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        w_g_n = w_g_mask.sum(dim=0)
        w_g_n[w_g_n == 0] = 1
        self.w_g_n = torch.nn.Parameter(w_g_n, requires_grad=False)  # (n, )
        self.w_p = torch.nn.Parameter(torch.zeros((p, n)).uniform_(-1, 1))  # (p, n)
        self.w_p_mask = torch.nn.Parameter(w_p_mask.any(dim=0), requires_grad=False)  # (p, n), bool
        w_p_n = w_p_mask.sum(dim=[0, 1])
        w_p_n[w_p_n == 0] = 1
        self.w_p_n = torch.nn.Parameter(w_p_n, requires_grad=False)  # (n, )
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.state_func = torch.nn.Hardtanh(-1, 1)
        self.activation_func = torch.nn.Sigmoid()
        self.activation_scaling = torch.sigmoid(torch.tensor([-1, 1])).tolist()
        self.tau_m = torch.nn.Parameter(torch.zeros(m).uniform_(0.04, 1))  # (m, )
        self.alpha_m = torch.nn.Parameter(torch.zeros(m).uniform_(0, 1))  # (m, )
        self.beta_m = torch.nn.Parameter(torch.zeros(m).uniform_(0, 1))  # (m, )

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        calcium = torch.zeros(self.m)
        return state, activation, calcium  # (n, ), (n, ), (m, )

    def _external_input(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
        return external_input

    def forward(self, state, activation, calcium, stimuli):
        # chemical synapse weight
        w_c = self.w_c.abs() * self.w_c_mask
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        dt_tau = self.dt / self.steps / self.tau.clamp(0.01, 0.2)
        for i in range(self.steps):
            # chemical synapse input
            synapse_input = (torch.mm(activation, w_c * self.e_c) - torch.mm(activation, w_c) * state) / self.w_c_n
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
        action = activation[:, self.output_index]  # muscle activation
        dt_tau_m = self.dt / self.tau_m.clamp(0.04, 1)
        calcium = (1 - dt_tau_m) * calcium + dt_tau_m * action  # calcium concentration
        f = self.alpha_m.abs() * calcium + self.beta_m.abs()  # calcium fluorescence
        return state, activation, calcium, f


class SNNCalciumFluorescence(torch.nn.Module):
    def __init__(self, cell):
        super(SNNCalciumFluorescence, self).__init__()
        self.cell = cell

    def init_state(self, batch_size, device):
        """ initial state and activation """
        state, activation, calcium = self.cell.init_state  # (n, ), (n, ), (m, )
        state = state.unsqueeze(dim=0).repeat(batch_size, 1).to(device=device)
        activation = activation.unsqueeze(dim=0).repeat(batch_size, 1).to(device=device)
        calcium = calcium.unsqueeze(dim=0).repeat(batch_size, 1).to(device=device)
        return state, activation, calcium

    def forward(self, stimuli):
        device = stimuli.device
        batch_size = stimuli.size(0)
        seq_len = stimuli.size(1)
        # initial state and activation
        state, activation, calcium = self.init_state(batch_size, device)
        # simulate sequence
        actions = []
        for t in range(seq_len):
            s = stimuli[:, t]
            state, activation, calcium, action = self.cell.forward(state, activation, calcium, s)
            actions.append(action)  # action, (batch_size, muscle_count)
        actions = torch.stack(actions, dim=1)  # action sequence, (batch_size, seq_len, muscle_count)
        return actions

    def step(self, state, activation, calcium, stimuli):
        if state is None or activation is None or calcium is None:
            device = stimuli.device
            batch_size = stimuli.size(0)
            state, activation, calcium = self.init_state(batch_size, device)
        state, activation, calcium, action = self.cell.forward(state, activation, calcium, stimuli)
        return state, activation, calcium, action
