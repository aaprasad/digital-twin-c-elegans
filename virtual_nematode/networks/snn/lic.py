import torch


class LIC(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC, self).__init__()
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
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        return state, activation  # (n, ), (n, )

    def _external_input_proprioception(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
        return external_input

    def _external_input(self, stimuli):
        """ receive proprioception input and sensory input
        https://doi.org/10.1038/nature06927
        https://doi.org/10.1038/s41598-018-35157-1
        """
        # proprioception input
        external_input = self._external_input_proprioception(stimuli[:, 0:self.p])
        # sensory input
        gradient = stimuli[:, self.p:self.p+1]  # (batch_size, 1)
        up_step_index = gradient > 0
        down_step_index = gradient <= 0
        w_s = self.w_s.abs()
        asel_input = torch.zeros_like(gradient)
        asel_input[up_step_index] = w_s[0] * gradient[up_step_index]
        aser_input = torch.zeros_like(gradient)
        aser_input[up_step_index] = -w_s[1] * gradient[up_step_index]
        aser_input[down_step_index] = -w_s[2] * gradient[down_step_index]
        sensory_input = torch.cat((asel_input, aser_input), dim=1)  # ASEL/ASER, (batch_size, 2)
        external_input[:, self.w_s_mask] += sensory_input
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
        return state, activation, action


class LIC1(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC1, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        # self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).normal_(mean=0, std=0.05))  # (n, )
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
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        return state, activation  # (n, ), (n, )

    def _external_input_proprioception(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
        return external_input

    def _external_input(self, stimuli):
        """ receive proprioception input and sensory input
        https://doi.org/10.1038/nature06927
        https://doi.org/10.1038/s41598-018-35157-1
        """
        # proprioception input
        external_input = self._external_input_proprioception(stimuli[:, 0:self.p])
        # sensory input
        gradient = stimuli[:, self.p:self.p+1]  # (batch_size, 1)
        up_step_index = gradient > 0
        down_step_index = gradient <= 0
        w_s = self.w_s.abs()
        asel_input = torch.zeros_like(gradient)
        asel_input[up_step_index] = w_s[0] * gradient[up_step_index]
        aser_input = torch.zeros_like(gradient)
        aser_input[up_step_index] = -w_s[1] * gradient[up_step_index]
        aser_input[down_step_index] = -w_s[2] * gradient[down_step_index]
        sensory_input = torch.cat((asel_input, aser_input), dim=1)  # ASEL/ASER, (batch_size, 2)
        external_input[:, self.w_s_mask] += sensory_input
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
        return state, activation, action


class LIC2(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask, beta=3.):
        super(LIC2, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        # self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).normal_(mean=0, std=0.05))  # (n, )
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
        self.activation_scaling = torch.sigmoid(torch.tensor([-1 * beta, 1 * beta])).tolist()
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long
        self.beta = beta

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(self.beta * state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        return state, activation  # (n, ), (n, )

    def _external_input_proprioception(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
        return external_input

    def _external_input(self, stimuli):
        """ receive proprioception input and sensory input
        https://doi.org/10.1038/nature06927
        https://doi.org/10.1038/s41598-018-35157-1
        """
        # proprioception input
        external_input = self._external_input_proprioception(stimuli[:, 0:self.p])
        # sensory input
        gradient = stimuli[:, self.p:self.p+1]  # (batch_size, 1)
        up_step_index = gradient > 0
        down_step_index = gradient <= 0
        w_s = self.w_s.abs()
        asel_input = torch.zeros_like(gradient)
        asel_input[up_step_index] = w_s[0] * gradient[up_step_index]
        aser_input = torch.zeros_like(gradient)
        aser_input[up_step_index] = -w_s[1] * gradient[up_step_index]
        aser_input[down_step_index] = -w_s[2] * gradient[down_step_index]
        sensory_input = torch.cat((asel_input, aser_input), dim=1)  # ASEL/ASER, (batch_size, 2)
        external_input[:, self.w_s_mask] += sensory_input
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
            activation = self.activation_func(self.beta * state)
            activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC3(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask, beta=3.):
        super(LIC3, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        # self.bias = torch.nn.Parameter(torch.zeros(n).uniform_(-1, 1))  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).normal_(mean=0, std=0.05))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.05)
        # e_c = (0 + e_c) * w_c_mask[0] + (1 + e_c) * w_c_mask[1] + (-1 + e_c) * w_c_mask[2]
        e_c = (0 + e_c) * w_c_mask[0] + (1 - e_c.abs()) * w_c_mask[1] + (-1 + e_c.abs()) * w_c_mask[2]
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
        self.activation_scaling = torch.sigmoid(torch.tensor([-1 * beta, 1 * beta])).tolist()
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long
        self.beta = beta
        self.reversal_potential_func = torch.nn.Hardtanh(-1, 1)

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(self.beta * state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        return state, activation  # (n, ), (n, )

    def _external_input_proprioception(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
        return external_input

    def _external_input(self, stimuli):
        """ receive proprioception input and sensory input
        https://doi.org/10.1038/nature06927
        https://doi.org/10.1038/s41598-018-35157-1
        """
        # proprioception input
        external_input = self._external_input_proprioception(stimuli[:, 0:self.p])
        # sensory input
        gradient = stimuli[:, self.p:self.p+1]  # (batch_size, 1)
        up_step_index = gradient > 0
        down_step_index = gradient <= 0
        w_s = self.w_s.abs()
        asel_input = torch.zeros_like(gradient)
        asel_input[up_step_index] = w_s[0] * gradient[up_step_index]
        aser_input = torch.zeros_like(gradient)
        aser_input[up_step_index] = -w_s[1] * gradient[up_step_index]
        aser_input[down_step_index] = -w_s[2] * gradient[down_step_index]
        sensory_input = torch.cat((asel_input, aser_input), dim=1)  # ASEL/ASER, (batch_size, 2)
        external_input[:, self.w_s_mask] += sensory_input
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
            activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC4(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC4, self).__init__()
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
        # e_c = (0 + e_c) * w_c_mask[0] + (1 + e_c) * w_c_mask[1] + (-1 + e_c) * w_c_mask[2]
        e_c = (0 + e_c) * w_c_mask[0] + (1 - e_c.abs()) * w_c_mask[1] + (-1 + e_c.abs()) * w_c_mask[2]
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
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long
        self.reversal_potential_func = torch.nn.Hardtanh(-1, 1)

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        return state, activation  # (n, ), (n, )

    def _external_input_proprioception(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
        return external_input

    def _external_input(self, stimuli):
        """ receive proprioception input and sensory input
        https://doi.org/10.1038/nature06927
        https://doi.org/10.1038/s41598-018-35157-1
        """
        # proprioception input
        external_input = self._external_input_proprioception(stimuli[:, 0:self.p])
        # sensory input
        gradient = stimuli[:, self.p:self.p+1]  # (batch_size, 1)
        up_step_index = gradient > 0
        down_step_index = gradient <= 0
        w_s = self.w_s.abs()
        asel_input = torch.zeros_like(gradient)
        asel_input[up_step_index] = w_s[0] * gradient[up_step_index]
        aser_input = torch.zeros_like(gradient)
        aser_input[up_step_index] = -w_s[1] * gradient[up_step_index]
        aser_input[down_step_index] = -w_s[2] * gradient[down_step_index]
        sensory_input = torch.cat((asel_input, aser_input), dim=1)  # ASEL/ASER, (batch_size, 2)
        external_input[:, self.w_s_mask] += sensory_input
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
            activation = self.activation_func(state)
            activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC5(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC5, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        self.bias = torch.nn.Parameter(torch.zeros(n).normal_(mean=0, std=0.05))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.05)
        # e_c = (0 + e_c) * w_c_mask[0] + (1 + e_c) * w_c_mask[1] + (-1 + e_c) * w_c_mask[2]
        e_c = (0 + e_c) * w_c_mask[0] + (1 - e_c.abs()) * w_c_mask[1] + (-1 + e_c.abs()) * w_c_mask[2]
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
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long
        self.reversal_potential_func = torch.nn.Hardtanh(-1, 1)

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        state = self.state_func(bias)
        activation = self.activation_func(state)
        activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        return state, activation  # (n, ), (n, )

    def _external_input_proprioception(self, stimuli):
        """ proprioception input """
        w_p = self.w_p * self.w_p_mask
        external_input = torch.mm(stimuli, w_p) / self.w_p_n
        return external_input

    def _external_input(self, stimuli):
        """ receive proprioception input and sensory input
        https://doi.org/10.1038/nature06927
        https://doi.org/10.1038/s41598-018-35157-1
        """
        # proprioception input
        external_input = self._external_input_proprioception(stimuli[:, 0:self.p])
        # sensory input
        gradient = stimuli[:, self.p:self.p+1]  # (batch_size, 1)
        up_step_index = gradient > 0
        down_step_index = gradient <= 0
        w_s = self.w_s.abs()
        asel_input = torch.zeros_like(gradient)
        asel_input[up_step_index] = w_s[0] * gradient[up_step_index]
        aser_input = torch.zeros_like(gradient)
        aser_input[up_step_index] = -w_s[1] * gradient[up_step_index]
        aser_input[down_step_index] = -w_s[2] * gradient[down_step_index]
        sensory_input = torch.cat((asel_input, aser_input), dim=1)  # ASEL/ASER, (batch_size, 2)
        external_input[:, self.w_s_mask] += sensory_input
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
            activation = self.activation_func(state)
            activation = (activation - self.activation_scaling[0]) / (self.activation_scaling[1] - self.activation_scaling[0])
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action
