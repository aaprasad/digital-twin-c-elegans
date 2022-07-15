import numpy as np
import pandas as pd
import torch
from virtual_nematode.networks.snn.forward import Connectome as _Connectome
from virtual_nematode.networks.snn.forward import SNNCell as _SNNCell


class Connectome(_Connectome):
    def __init__(self, gradient_size, gradient_mask, **kwargs):
        super(Connectome, self).__init__(**kwargs)
        self.gradient_size = gradient_size
        self.gradient_mask = gradient_mask

    def _gradient_mask(self):
        mask = pd.DataFrame(False, index=list(range(self.gradient_size)), columns=self.cells)
        if self.gradient_mask is True:  # sensory neurons receive gradient input
            mask.loc[:, self.sensory_neurons] = True
        else:  # all neurons receive gradient input
            mask.loc[:, self.neurons] = True
        mask = torch.from_numpy(mask.to_numpy(dtype=np.bool))
        return mask

    def mask(self):
        masks = super(Connectome, self).mask()
        w_gradient_mask = self._gradient_mask()
        return masks, w_gradient_mask


class SNNCell(_SNNCell):
    def __init__(self, gradient_size, w_gradient_mask, **kwargs):
        super(SNNCell, self).__init__(**kwargs)
        self.gradient_size = gradient_size
        n = kwargs.get('n')
        self.w_gradient = torch.nn.Parameter(torch.zeros((gradient_size, n)).uniform_(-1, 1))  # gradient input synapse weight, (gradient_size, cell_count)
        self.w_gradient_mask = torch.nn.Parameter(w_gradient_mask, requires_grad=False)  # gradient input synapse bool mask, (gradient_size, cell_count)
        w_gradient_n = w_gradient_mask.sum(dim=0)
        w_gradient_n[w_gradient_n == 0] = 1
        self.w_gradient_n = torch.nn.Parameter(w_gradient_n, requires_grad=False)  # gradient input synapse amount, (cell_count, )

    def _external_input(self, stimuli):
        """ external input
        stimuli: (batch_size, proprioception_size + gradient_size)
        """
        # proprioception input, (batch_size, cell_count)
        external_input = super(SNNCell, self)._external_input(stimuli[:, 0:self.p])
        # gradient input, (batch_size, cell_count)
        w_gradient = self.w_gradient * self.w_gradient_mask
        external_input += torch.mm(stimuli[:, self.p:self.p+self.gradient_size], w_gradient) / self.w_gradient_n
        return external_input
