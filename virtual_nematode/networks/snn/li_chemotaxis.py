import torch
from virtual_nematode.networks.snn.li import LI0, LI1


class LIG0(LI0):
    def __init__(self, s, w_s_mask, **kwargs):
        super(LIG0, self).__init__(**kwargs)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    def _external_input(self, stimuli):
        """ receive proprioception input and sensory input
        https://doi.org/10.1038/nature06927
        https://doi.org/10.1038/s41598-018-35157-1
        """
        # proprioception input
        external_input = super(LIG0, self)._external_input(stimuli[:, 0:self.p])
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


class LIG1(LI1):
    def __init__(self, s, w_s_mask, **kwargs):
        super(LIG1, self).__init__(**kwargs)
        self.s = s  # sensory size
        self.w_s = torch.nn.Parameter(torch.ones(s))  # (3, )
        self.w_s_mask = torch.nn.Parameter(w_s_mask, requires_grad=False)  # (2, ), long

    def _external_input(self, stimuli):
        """ receive proprioception input and sensory input
        https://doi.org/10.1038/nature06927
        https://doi.org/10.1038/s41598-018-35157-1
        """
        # proprioception input
        external_input = super(LIG1, self)._external_input(stimuli[:, 0:self.p])
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
