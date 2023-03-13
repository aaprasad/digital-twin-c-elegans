import torch
from virtual_nematode.networks.snn.output import MuscleArrangement


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


class LIC11(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC11, self).__init__()
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
        self.muscle_arrangement = MuscleArrangement(band_lengths=[24, 24, 23, 24], kernel_size=3)

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
        action = self.muscle_arrangement(action)
        return state, activation, action


class Activation(torch.nn.Module):
    def __init__(self, k=12.5, b=3.):
        super(Activation, self).__init__()
        self.k = k
        self.b = b

    def forward(self, input):
        return torch.sigmoid(self.k * input + self.b)


class LIC20(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask, w_max=1.):
        super(LIC20, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        self.bias = torch.nn.Parameter(-0.35 + torch.zeros(n).normal_(0, 0.01))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
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
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long
        self.w_max = w_max

    @property
    def init_state(self):
        """ initial state and activation """
        state = torch.zeros(self.n).fill_(-0.35)
        activation = self.activation_func(state)
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
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
        w_c = self.w_c.clamp(0., self.w_max) * self.w_c_mask
        e_c = self.e_c.clamp(-0.5, 0.05)
        # gap junction weight
        w_g = self.w_g.clamp(0., self.w_max)
        # w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        bias = self.bias.clamp(-0.45, 0.)
        external_input = self._external_input(stimuli) + bias
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
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC21(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask, w_max=1.):
        super(LIC21, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        self.bias = torch.nn.Parameter(-0.35 + torch.zeros(n).normal_(0, 0.01))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
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
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long
        self.w_max = w_max

    @property
    def init_state(self):
        """ initial state and activation """
        bias = self.bias.clone().detach()
        bias = bias.clamp(-0.45, 0.)
        state = bias
        activation = self.activation_func(state)
        return state, activation  # (n, ), (n, )

    def _external_input_proprioception(self, stimuli):
        """ proprioception input """
        w_p = self.w_p.clamp(-self.w_max, self.w_max) * self.w_p_mask
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
        up_step_index = gradient > 0
        down_step_index = gradient <= 0
        w_s = self.w_s.clamp(0., self.w_max)
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
        w_c = self.w_c.clamp(0., self.w_max) * self.w_c_mask
        e_c = self.e_c.clamp(-0.5, 0.05)
        # gap junction weight
        w_g = self.w_g.clamp(0., 1.)
        # w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        bias = self.bias.clamp(-0.45, 0.)
        external_input = self._external_input(stimuli) + bias
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
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC22(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask, w_max=1.):
        super(LIC22, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        # tau
        tau_clip = SigmoidClamp(a=0.01, b=0.2)
        tau = torch.zeros(n).uniform_(0.01, 0.2)
        tau = tau_clip.inverse(tau)
        self.tau = torch.nn.Parameter(tau)  # (n, )
        self.tau_clip = tau_clip
        # bias
        bias_clip = SigmoidClamp(a=-0.45, b=0.)
        bias = -0.35 + torch.zeros(n).normal_(0, 0.01)
        bias = bias_clip.inverse(bias)
        self.bias = torch.nn.Parameter(bias)  # (n, )
        self.bias_clip = bias_clip
        # w_c
        w_c_clip = SigmoidClamp(a=0., b=w_max)
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c = w_c_clip.inverse(w_c)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        self.w_c_clip = w_c_clip
        # e_c
        e_c_clip = SigmoidClamp(a=-0.5, b=0.05)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
        e_c = e_c_clip.inverse(e_c)
        self.e_c = torch.nn.Parameter(e_c)  # (n, n)
        self.e_c_clip = e_c_clip
        self.w_c_mask = torch.nn.Parameter(w_c_mask.any(dim=0), requires_grad=False)  # (n, n), bool
        w_c_n = w_c_mask.sum(dim=[0, 1])
        w_c_n[w_c_n == 0] = 1
        self.w_c_n = torch.nn.Parameter(w_c_n, requires_grad=False)  # (n, )
        # w_g
        w_g_clip = SigmoidClamp(a=0., b=w_max)
        w_g = torch.zeros((n, n)).uniform_(0, 1)
        w_g = w_g_clip.inverse(w_g)
        self.w_g = torch.nn.Parameter(w_g)  # (n, n)
        self.w_g_clip = w_g_clip
        self.w_g_mask = torch.nn.Parameter(w_g_mask, requires_grad=False)  # (n, n), bool
        w_g_n = w_g_mask.sum(dim=0)
        w_g_n[w_g_n == 0] = 1
        self.w_g_n = torch.nn.Parameter(w_g_n, requires_grad=False)  # (n, )
        # w_p
        # w_p_clip = SigmoidClamp(a=-w_max, b=w_max)
        w_p = torch.zeros((p, n)).uniform_(-1, 1)
        # w_p = w_p_clip.inverse(w_p)
        self.w_p = torch.nn.Parameter(w_p)  # (p, n)
        # self.w_p_clip = w_p_clip
        self.w_p_mask = torch.nn.Parameter(w_p_mask.any(dim=0), requires_grad=False)  # (p, n), bool
        w_p_n = w_p_mask.sum(dim=[0, 1])
        w_p_n[w_p_n == 0] = 1
        self.w_p_n = torch.nn.Parameter(w_p_n, requires_grad=False)  # (n, )
        self.output_index = torch.nn.Parameter(output_index, requires_grad=False)  # (n, ), bool
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        # w_s
        # w_s_clip = SigmoidClamp(a=0., b=50.)
        w_s = torch.ones(s)
        # w_s = w_s_clip.inverse(w_s)
        self.w_s = torch.nn.Parameter(w_s)  # (3, )
        # self.w_s_clip = w_s_clip
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long
        self.w_max = w_max

    @property
    def init_state(self):
        """ initial state and activation """
        state = torch.zeros(self.n).fill_(-0.35)
        activation = self.activation_func(state)
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
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
        w_c = self.w_c_clip(self.w_c) * self.w_c_mask
        e_c = self.e_c_clip(self.e_c)
        # gap junction weight
        w_g = self.w_g_clip(self.w_g)
        # w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        bias = self.bias_clip(self.bias)
        external_input = self._external_input(stimuli) + bias
        # dt / tau
        tau = self.tau_clip(self.tau)
        dt_tau = self.dt / self.steps / tau
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
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC23(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask, w_max=1.):
        super(LIC23, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        self.bias = torch.nn.Parameter(-0.35 + torch.zeros(n).normal_(0, 0.01))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
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
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long
        self.w_max = w_max

    @property
    def init_state(self):
        """ initial state and activation """
        state = torch.zeros(self.n).fill_(-0.35)
        activation = self.activation_func(state)
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
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
        w_c = self.w_c.clamp(0., self.w_max) * self.w_c_mask
        e_c = self.e_c.clamp(-0.5, 0.05)
        # gap junction weight
        w_g = self.w_g.clamp(0., self.w_max)
        # w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        bias = self.bias.clamp(-0.45, -0.24)
        external_input = self._external_input(stimuli) + bias
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
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC30(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC30, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        self.bias = torch.nn.Parameter(-0.35 + torch.zeros(n).normal_(0, 0.01))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
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
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    @property
    def init_state(self):
        """ initial state and activation """
        state = torch.zeros(self.n).fill_(-0.35)
        activation = self.activation_func(state)
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
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
        e_c = self.e_c.clamp(-0.5, 0.05)
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
            state = state.clamp(-0.8, 0.35)
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC31(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC31, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        self.bias = torch.nn.Parameter(-0.35 + torch.zeros(n).normal_(0, 0.01))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
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
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    @property
    def init_state(self):
        """ initial state and activation """
        state = torch.zeros(self.n).fill_(-0.35)
        activation = self.activation_func(state)
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
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
        e_c = self.e_c.clamp(-0.5, 0.05)
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
            total_input = total_input.clamp(-0.8, 0.35)
            # cell state and activation
            state = (1 - dt_tau) * state + dt_tau * total_input
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class SigmoidClamp(torch.nn.Module):
    def __init__(self, a, b):
        super(SigmoidClamp, self).__init__()
        self.a = a
        self.b = b
        self.k = b - a

    def inverse(self, input):
        e = 1e-6
        input = input.clamp(self.a + e, self.b - e)
        return torch.logit((input - self.a) / self.k)

    def forward(self, input):
        return self.k * torch.sigmoid(input) + self.a


class LIC40(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC40, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        self.bias = torch.nn.Parameter(-0.35 + torch.zeros(n).normal_(0, 0.01))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
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
        self.state_func = SigmoidClamp(a=-0.8, b=0.35)
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    @property
    def init_state(self):
        """ initial state and activation """
        state = torch.zeros(self.n).fill_(-0.35)
        activation = self.activation_func(state)
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
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
        e_c = self.e_c.clamp(-0.5, 0.05)
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
            total_input = self.input_func(total_input)
            # cell state and activation
            state = (1 - dt_tau) * state + dt_tau * total_input
            state = self.state_func(state)
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC41(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC41, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        tau = torch.zeros(n).uniform_(0.01, 0.2)
        tau_func = SigmoidClamp(a=0.01, b=0.2)
        tau = tau_func.inverse(tau)
        self.tau = torch.nn.Parameter(tau)  # (n, )
        self.tau_func = tau_func
        self.bias = torch.nn.Parameter(-0.35 + torch.zeros(n).normal_(0, 0.01))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
        e_c_func = SigmoidClamp(a=-0.5, b=0.05)
        e_c = e_c_func.inverse(e_c)
        self.e_c = torch.nn.Parameter(e_c)  # (n, n)
        self.e_c_func = e_c_func
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
        self.state_func = SigmoidClamp(a=-0.8, b=0.35)
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    @property
    def init_state(self):
        """ initial state and activation """
        state = torch.zeros(self.n).fill_(-0.35)
        activation = self.activation_func(state)
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
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
        e_c = self.e_c_func(self.e_c)
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        tau = self.tau_func(self.tau)
        dt_tau = self.dt / self.steps / tau
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
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC42(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC42, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        self.tau = torch.nn.Parameter(torch.zeros(n).uniform_(0.01, 0.2))  # (n, )
        self.bias = torch.nn.Parameter(-0.35 + torch.zeros(n).normal_(0, 0.01))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
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
        self.input_func = SigmoidClamp(a=-0.8, b=0.35)
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    @property
    def init_state(self):
        """ initial state and activation """
        state = torch.zeros(self.n).fill_(-0.35)
        activation = self.activation_func(state)
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
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
        e_c = self.e_c.clamp(-0.5, 0.05)
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
            total_input = self.input_func(total_input)
            # cell state and activation
            state = (1 - dt_tau) * state + dt_tau * total_input
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC43(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC43, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        tau = torch.zeros(n).uniform_(0.01, 0.2)
        tau_func = SigmoidClamp(a=0.01, b=0.2)
        tau = tau_func.inverse(tau)
        self.tau = torch.nn.Parameter(tau)  # (n, )
        self.tau_func = tau_func
        self.bias = torch.nn.Parameter(-0.35 + torch.zeros(n).normal_(0, 0.01))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
        e_c_func = SigmoidClamp(a=-0.5, b=0.05)
        e_c = e_c_func.inverse(e_c)
        self.e_c = torch.nn.Parameter(e_c)  # (n, n)
        self.e_c_func = e_c_func
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
        self.input_func = SigmoidClamp(a=-0.8, b=0.35)
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    @property
    def init_state(self):
        """ initial state and activation """
        state = torch.zeros(self.n).fill_(-0.35)
        activation = self.activation_func(state)
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
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
        e_c = self.e_c_func(self.e_c)
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        tau = self.tau_func(self.tau)
        dt_tau = self.dt / self.steps / tau
        for i in range(self.steps):
            # chemical synapse input
            synapse_input = (torch.mm(activation, w_c * e_c) - torch.mm(activation, w_c) * state) / self.w_c_n
            # gap junction input
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1) / self.w_g_n
            # total input
            total_input = synapse_input + gap_input + external_input
            total_input = self.input_func(total_input)
            # cell state and activation
            state = (1 - dt_tau) * state + dt_tau * total_input
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action


class LIC50(torch.nn.Module):
    def __init__(self, dt, steps, n, m, p, w_c_mask, w_g_mask, w_p_mask, output_index, s, w_s_mask):
        super(LIC50, self).__init__()
        self.dt = dt
        self.steps = steps
        self.n = n
        self.m = m
        self.p = p
        tau = torch.zeros(n).uniform_(0.01, 0.2)
        tau_func = SigmoidClamp(a=0.01, b=0.2)
        tau = tau_func.inverse(tau)
        self.tau = torch.nn.Parameter(tau)  # (n, )
        self.tau_func = tau_func
        self.bias = torch.nn.Parameter(-0.35 + torch.zeros(n).normal_(0, 0.01))  # (n, )
        w_c = torch.zeros((n, n)).uniform_(0, 1)
        w_c *= w_c_mask.any(dim=0)
        self.w_c = torch.nn.Parameter(w_c)  # (n, n)
        e_c = torch.zeros((n, n)).normal_(mean=0, std=0.01)
        e_c = (-0.24 + e_c) * w_c_mask[0] + (0. + e_c) * w_c_mask[1] + (-0.45 + e_c) * w_c_mask[2]
        e_c_func = SigmoidClamp(a=-0.5, b=0.05)
        e_c = e_c_func.inverse(e_c)
        self.e_c = torch.nn.Parameter(e_c)  # (n, n)
        self.e_c_func = e_c_func
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
        self.input_func = SigmoidClamp(a=-15, b=15)  # if tau=0.08, dt/tau*input~(-1.5,1.5), unit 0.1 V
        self.activation_func = Activation(k=37.5, b=9.)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    @property
    def init_state(self):
        """ initial state and activation """
        state = torch.zeros(self.n).fill_(-0.35)
        activation = self.activation_func(state)
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
        gradient = stimuli[:, self.p:self.p + 1]  # (batch_size, 1)
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
        e_c = self.e_c_func(self.e_c)
        # gap junction weight
        w_g = self.w_g.abs()
        w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * self.w_g_mask
        # external input + bias
        external_input = self._external_input(stimuli) + self.bias
        # dt / tau
        tau = self.tau_func(self.tau)
        dt_tau = self.dt / self.steps / tau
        for i in range(self.steps):
            # chemical synapse input
            synapse_input = (torch.mm(activation, w_c * e_c) - torch.mm(activation, w_c) * state) / self.w_c_n
            # gap junction input
            delta_state = state.unsqueeze(dim=2).repeat(1, 1, self.n) - state.unsqueeze(dim=1).repeat(1, self.n, 1)
            gap_input = torch.sum(delta_state * w_g, dim=1) / self.w_g_n
            # total input
            total_input = synapse_input + gap_input + external_input
            total_input = self.input_func(total_input)
            # cell state and activation
            state = (1 - dt_tau) * state + dt_tau * total_input
            activation = self.activation_func(state)
        # muscle output
        action = activation[:, self.output_index]
        return state, activation, action
