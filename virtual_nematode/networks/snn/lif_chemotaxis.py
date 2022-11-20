import torch
from virtual_nematode.networks.snn.lif import LeakyIntegratorConductanceBased


class LeakyIntegratorConductanceBasedGradientInput(LeakyIntegratorConductanceBased):
    def __init__(self, s, w_s_mask, **kwargs):
        super(LeakyIntegratorConductanceBasedGradientInput, self).__init__(**kwargs)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (s, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (s, ), long

    def _external_input(self, stimuli):
        """ receive proprioception input and sensory input
        https://doi.org/10.1038/nature06927
        https://doi.org/10.1038/s41598-018-35157-1
        """
        # proprioception input
        external_input = super(LeakyIntegratorConductanceBasedGradientInput, self)._external_input(stimuli[:, 0:self.p])
        # sensory input
        gradient = stimuli[:, self.p:self.p+1]  # (batch_size, 1)
        sensory_input = torch.cat((gradient.clamp(min=0), -gradient), dim=1)  # ASEL/ASER, (batch_size, s)
        external_input[:, self.w_s_mask] += sensory_input * self.w_s.abs()
        return external_input


class LeakyIntegratorConductanceBasedGradientInput1(LeakyIntegratorConductanceBased):
    def __init__(self, s, w_s_mask, **kwargs):
        super(LeakyIntegratorConductanceBasedGradientInput1, self).__init__(**kwargs)
        self.s = s  # sensory size
        self.w_s_a = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_b = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    def _external_input(self, stimuli):
        """ receive proprioception input and sensory input
        https://doi.org/10.1038/nature06927
        https://doi.org/10.1038/s41598-018-35157-1
        """
        # proprioception input
        external_input = super(LeakyIntegratorConductanceBasedGradientInput1, self)._external_input(stimuli[:, 0:self.p])
        # sensory input
        gradient = stimuli[:, self.p:self.p+1]  # (batch_size, 1)
        up_step_index = gradient > 0
        down_step_index = gradient <= 0
        asel_up_step_input = self.w_s_a[0].abs() * torch.log(self.w_s_b[0].abs() * gradient[up_step_index] + 1)
        aser_up_step_input = -self.w_s_a[1].abs() * torch.log(self.w_s_b[1].abs() * gradient[up_step_index] + 1)
        aser_down_step_input = self.w_s_a[2].abs() * torch.log(-self.w_s_b[2].abs() * gradient[down_step_index] + 1)
        asel_input = torch.zeros_like(gradient)
        asel_input[up_step_index] = asel_up_step_input
        aser_input = torch.zeros_like(gradient)
        aser_input[up_step_index] = aser_up_step_input
        aser_input[down_step_index] = aser_down_step_input
        sensory_input = torch.cat((asel_input, aser_input), dim=1)  # ASEL/ASER, (batch_size, 2)
        external_input[:, self.w_s_mask] += sensory_input
        return external_input
