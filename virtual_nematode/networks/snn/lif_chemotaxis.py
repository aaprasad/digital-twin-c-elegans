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
