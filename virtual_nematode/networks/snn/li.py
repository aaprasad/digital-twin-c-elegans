import torch


class LI0(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, v_range, scaling, init_type):
        """ trainable reversal potential
        v_range: (v_min, v_max)
        scaling: True or False
        init_type: 'random' or 'polarity'
        """
        super(LI0, self).__init__()
        self.dt = dt  # 0.04
        self.steps = steps  # 5
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).normal_(0, 1))  # (n, )
        w_c = torch.zeros((n, n)).normal_(0, 1)
        w_c = w_c.abs() * w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        self.w_c_mask = torch.nn.Parameter(w_c_mask.any(dim=0), requires_grad=False)  # (n, n), bool
        e_c = torch.zeros((n, n))
        torch.nn.init.trunc_normal_(e_c, mean=0, std=1, a=v_range[0], b=v_range[1])
        if init_type == 'random':
            e_c *= w_c_mask.any(dim=0)
        elif init_type == 'polarity':
            error = torch.zeros((n, n)).normal_(0, 1)
            e_c = e_c * w_c_mask[0] + (v_range[1] - error.abs()) * w_c_mask[1] + (v_range[0] + error.abs()) * w_c_mask[2]
        else:
            raise AssertionError
        self.e_c = torch.nn.Parameter(e_c)  # (n, n)
        w_c_n = w_c_mask.sum(dim=[0, 1])
        w_c_n[w_c_n == 0] = 1
        self.w_c_n = torch.nn.Parameter(w_c_n, requires_grad=False)  # (n, )
        w_g = torch.zeros((n, n)).normal_(0, 1)
        w_g = w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * w_g_mask
        self.w_g = torch.nn.Parameter(w_g)  # (n, n)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        w_g_n = w_g_mask.sum(dim=0)
        w_g_n[w_g_n == 0] = 1
        self.w_g_n = torch.nn.Parameter(w_g_n, requires_grad=False)  # (n, )
        w_p = torch.zeros((p, n)).normal_(0, 1)
        w_p *= w_p_mask.any(dim=0)
        self.w_p = torch.nn.Parameter(w_p)  # (p, n)
        self.w_p_mask = torch.nn.Parameter(w_p_mask.any(dim=0), requires_grad=False)  # (p, n), bool
        w_p_n = w_p_mask.sum(dim=[0, 1])
        w_p_n[w_p_n == 0] = 1
        self.w_p_n = torch.nn.Parameter(w_p_n, requires_grad=False)  # (n, )
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.state_func = torch.nn.Hardtanh(v_range[0], v_range[1])
        self.activation_func = torch.nn.Sigmoid()
        self.v_range = v_range
        self.scaling = scaling
        self.activation_scaling = torch.sigmoid(torch.tensor(v_range)).tolist()

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(state)
        if self.scaling is True:
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
        e_c = self.e_c.clamp(self.v_range[0], self.v_range[1])
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
            if self.scaling is True:
                activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LI1(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, v_range, scaling):
        """ partially fixed reversal potential
        v_range: (v_min, v_max)
        """
        super(LI1, self).__init__()
        self.dt = dt  # 0.04
        self.steps = steps  # 5
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).normal_(0, 1))  # (n, )
        w_c = torch.zeros((n, n)).normal_(0, 1)
        w_c = w_c.abs() * w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        self.w_c_mask = torch.nn.Parameter(w_c_mask.any(dim=0), requires_grad=False)  # (n, n), bool
        e_c = torch.zeros((n, n))
        torch.nn.init.trunc_normal_(e_c, mean=0, std=1, a=v_range[0], b=v_range[1])
        e_c = e_c * w_c_mask[0] + v_range[1] * w_c_mask[1] + v_range[0] * w_c_mask[2]
        self.e_c = torch.nn.Parameter(e_c)  # (n, n)
        self.e_c_mask = torch.nn.Parameter(w_c_mask, requires_grad=False)  # (n, n), bool
        w_c_n = w_c_mask.sum(dim=[0, 1])
        w_c_n[w_c_n == 0] = 1
        self.w_c_n = torch.nn.Parameter(w_c_n, requires_grad=False)  # (n, )
        w_g = torch.zeros((n, n)).normal_(0, 1)
        w_g = w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * w_g_mask
        self.w_g = torch.nn.Parameter(w_g)  # (n, n)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        w_g_n = w_g_mask.sum(dim=0)
        w_g_n[w_g_n == 0] = 1
        self.w_g_n = torch.nn.Parameter(w_g_n, requires_grad=False)  # (n, )
        w_p = torch.zeros((p, n)).normal_(0, 1)
        w_p *= w_p_mask.any(dim=0)
        self.w_p = torch.nn.Parameter(w_p)  # (p, n)
        self.w_p_mask = torch.nn.Parameter(w_p_mask.any(dim=0), requires_grad=False)  # (p, n), bool
        w_p_n = w_p_mask.sum(dim=[0, 1])
        w_p_n[w_p_n == 0] = 1
        self.w_p_n = torch.nn.Parameter(w_p_n, requires_grad=False)  # (n, )
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.state_func = torch.nn.Hardtanh(v_range[0], v_range[1])
        self.activation_func = torch.nn.Sigmoid()
        self.v_range = v_range
        self.scaling = scaling
        self.activation_scaling = torch.sigmoid(torch.tensor(v_range)).tolist()

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(state)
        if self.scaling is True:
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
        e_c = self.e_c.clamp(self.v_range[0], self.v_range[1]) * self.e_c_mask[0] + self.v_range[1] * self.e_c_mask[1] + self.v_range[0] * self.e_c_mask[2]
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
            if self.scaling is True:
                activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LI2(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, init_type, beta, v_range, e_range):
        """ trainable reversal potential
        init_type: 'random' or 'polarity'
        beta: the width of the sigmoid activation function
        v_range: range of membrane potential (v_min, v_max)
        e_range: range of reversal potential (e_min, e_max)
        """
        super(LI2, self).__init__()
        self.dt = dt  # 0.04
        self.steps = steps  # 5
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        bias = torch.zeros(n)
        torch.nn.init.trunc_normal_(bias, mean=0, std=1, a=e_range[0], b=e_range[1])
        self.bias = torch.nn.Parameter(bias)  # (n, )
        w_c = torch.zeros((n, n)).normal_(0, 1)
        w_c = w_c.abs() * w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        self.w_c_mask = torch.nn.Parameter(w_c_mask.any(dim=0), requires_grad=False)  # (n, n), bool
        e_c = torch.zeros((n, n))
        torch.nn.init.trunc_normal_(e_c, mean=0, std=1, a=e_range[0], b=e_range[1])
        if init_type == 'random':
            e_c *= w_c_mask.any(dim=0)
        elif init_type == 'polarity':
            error = torch.zeros((n, n)).normal_(0, 1)
            e_c = e_c * w_c_mask[0] + (e_range[1] - error.abs()) * w_c_mask[1] + (e_range[0] + error.abs()) * w_c_mask[2]
        else:
            raise AssertionError
        self.e_c = torch.nn.Parameter(e_c)  # (n, n)
        w_c_n = w_c_mask.sum(dim=[0, 1])
        w_c_n[w_c_n == 0] = 1
        self.w_c_n = torch.nn.Parameter(w_c_n, requires_grad=False)  # (n, )
        w_g = torch.zeros((n, n)).normal_(0, 1)
        w_g = w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * w_g_mask
        self.w_g = torch.nn.Parameter(w_g)  # (n, n)
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        w_g_n = w_g_mask.sum(dim=0)
        w_g_n[w_g_n == 0] = 1
        self.w_g_n = torch.nn.Parameter(w_g_n, requires_grad=False)  # (n, )
        w_p = torch.zeros((p, n)).normal_(0, 1)
        w_p *= w_p_mask.any(dim=0)
        self.w_p = torch.nn.Parameter(w_p)  # (p, n)
        self.w_p_mask = torch.nn.Parameter(w_p_mask.any(dim=0), requires_grad=False)  # (p, n), bool
        w_p_n = w_p_mask.sum(dim=[0, 1])
        w_p_n[w_p_n == 0] = 1
        self.w_p_n = torch.nn.Parameter(w_p_n, requires_grad=False)  # (n, )
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.state_func = torch.nn.Hardtanh(v_range[0], v_range[1])
        self.activation_func = torch.nn.Sigmoid()
        self.beta = beta
        self.reversal_potential_func = torch.nn.Hardtanh(e_range[0], e_range[1])

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(self.beta * state)
        return state, activation  # (n, ), (n, )

    def _external_input(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
        return external_input

    def forward(self, state, activation, stimuli):
        # chemical synapse weight
        w_c = self.w_c.abs() * self.w_c_mask
        e_c = self.reversal_potential_func(self.e_c)
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
            activation = self.activation_func(self.beta * state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action
