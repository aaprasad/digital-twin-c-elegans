import numpy as np
import torch


class NCPDataset(torch.utils.data.TensorDataset):
    """ prepare dataset for NCP training
    encode the sensory inputs and motor outputs of a chemotaxis TensorDataset
    split each sequence into subsequences
    x: encoded and normalized sensory inputs to ASEL/R
    y: normalized motor outputs
    """
    def __init__(self, dataset, seq_len):
        """ preprocess a TensorDataset for NCP network training """
        x, y = dataset.tensors
        x = x[:, 1:] - x[:, 0:-1]
        y = y[:, 1:, :]
        self.seq_len = seq_len
        self.x_abs_max = x.abs().max()
        self.y_abs_max = y.abs().max()
        x = self.encode_input_func(g=x)
        x = self.subsequence(x)
        y = self.subsequence(y)
        super(NCPDataset, self).__init__(x, y)

    def encode_input_func(self, g):
        """ encode input: gradient -> ASEL/R input
        ASEL/R's functional asymmetry
            ASEL: excited by NaCl increase, not affected by NaCl decrease
            ASER: inhibited by NaCl increase, excited by NaCl decrease
            https://doi.org/10.1038/nature06927
        log(*): gradient -> fluorescence response as input
            https://doi.org/10.1038/s41598-018-35157-1
        """
        # normalized gradient: [-1, 1]
        g /= self.x_abs_max
        g_abs_max = 1.

        # g > 0: ASEL excited
        b_l_plus = 100.
        a_l_plus = 1. / np.log(b_l_plus * g_abs_max + 1.)
        x_asel_plus = a_l_plus * torch.log(b_l_plus * g + 1.)
        # g > 0: ASER inhibited, but weaker than excitation response
        b_r_plus = 100.
        a_r_plus = -1. / np.log(b_r_plus * g_abs_max + 1.) / 2.
        x_aser_plus = a_r_plus * torch.log(b_r_plus * g + 1.)

        # g <= 0: ASEL not affected
        x_asel_minus = torch.zeros_like(g)
        # g <= 0: ASER excited
        b_r_minus = -100.
        a_r_minus = 1. / np.log(b_r_minus * -g_abs_max + 1.)
        x_aser_minus = a_r_minus * torch.log(b_r_minus * g + 1.)

        # ASEL/R response as input
        x_asel = torch.where(g > 0., x_asel_plus, x_asel_minus)
        x_aser = torch.where(g > 0., x_aser_plus, x_aser_minus)

        # because of coef a, x is normalized in [-1, 1]
        x = torch.stack([x_asel, x_aser], dim=-1)
        return x

    def subsequence(self, tensor):
        """ split each sequence into subsequences """
        tensor = tensor.split(self.seq_len, dim=1)
        if tensor[-1].size(1) != self.seq_len:
            tensor = tensor[:-1]
        tensor = torch.cat(tensor, dim=0)
        return tensor
