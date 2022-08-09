from virtual_nematode.connectomes.cells import body_wall_muscles, neuron_list2, sensory_neurons
from virtual_nematode.connectomes.forward import Connectome as _Connectome


class Connectome(_Connectome):
    def __init__(self, gradient_size, **kwargs):
        super(Connectome, self).__init__(**kwargs)
        self.gradient_size = gradient_size

    def _gradient_mask(self):
        mask = self._external_input_mask(dim=self.gradient_size)
        return mask

    def mask(self):
        masks = super(Connectome, self).mask()
        w_gradient_mask = self._gradient_mask()
        return masks, w_gradient_mask


def get_kwargs(path):
    """ connectome masks and params """
    muscles = body_wall_muscles()
    neurons = neuron_list2(path, muscles)
    sensory = sensory_neurons(path)
    print('{} neurons, {} muscles, {} sensory, {} cells in total'.format(len(neurons), len(muscles), len(sensory), len(neurons) + len(muscles)))
    p = 24
    gradient_size = 1
    connectome = Connectome(
        gradient_size=gradient_size, gradient_mask=True,
        path=path, neurons=neurons, muscles=muscles, ex_synapses=[], in_synapses=[],
        sensory_neurons=sensory, p=p
    )
    (w_c_mask, w_g_mask, w_p_mask, output_index), w_gradient_mask = connectome.mask()
    return {
        'n': len(connectome.cells), 'm': len(connectome.muscles), 'p': p,
        'w_c_mask': w_c_mask, 'w_g_mask': w_g_mask, 'w_p_mask': w_p_mask, 'output_index': output_index,
        'gradient_size': gradient_size, 'w_gradient_mask': w_gradient_mask,
    }
