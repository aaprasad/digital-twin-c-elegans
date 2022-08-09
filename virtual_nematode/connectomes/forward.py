import numpy as np
import pandas as pd
import torch
from virtual_nematode.connectomes.cells import body_wall_muscles, neuron_list2, sensory_neurons


class Connectome(object):
    def __init__(self, path, neurons, muscles, ex_synapses, in_synapses, sensory_neurons, p):
        self.path = path
        self.neurons = neurons
        self.muscles = muscles
        self.cells = neurons + muscles  # cells to simulate
        self.ex_synapses = self._check_synapses(ex_synapses)
        self.in_synapses = self._check_synapses(in_synapses)
        self.sensory_neurons = self._check_cells(sensory_neurons)
        self.p = p
        self.chemical, self.gap_junction = self._init()

    def _check_synapses(self, synapses):
        """ check if the cells from synapses exist, and make sure no duplicate synapses """
        synapse_cells = set([pre for pre, _ in synapses] + [post for _, post in synapses])
        assert synapse_cells.issubset(set(self.cells))
        assert len(synapses) == len(set(synapses))
        return synapses

    def _check_cells(self, cells):
        """ check if the cells exist """
        assert set(cells).issubset(set(self.cells))
        return cells

    def _init(self):
        chemical = pd.read_excel(self.path, sheet_name='hermaphrodite chemical', header=2, index_col=2).iloc[:300, 2:456]
        gap_junction = pd.read_excel(self.path, sheet_name='hermaphrodite gap jn symmetric', header=2, index_col=2).iloc[:469, 2:471]
        self._check(gap_junction)
        chemical = self._add(chemical)
        gap_junction = self._add(gap_junction)
        chemical = self._slice(chemical)
        gap_junction = self._slice(gap_junction)
        return chemical, gap_junction

    def _check(self, gap_junction):
        """ check if all the cells are included in the connectome """
        connectome_cells = set(list(gap_junction.index))
        assert set(self.cells).issubset(connectome_cells)

    def _add(self, adjacency):
        for cell in self.cells:
            if cell not in adjacency.index:
                adjacency.loc[cell, :] = np.full_like(adjacency.columns, fill_value=np.nan)
            if cell not in adjacency.columns:
                adjacency.loc[:, cell] = np.full_like(adjacency.index, fill_value=np.nan)
        return adjacency

    def _slice(self, adjacency):
        adjacency = adjacency.loc[self.cells, self.cells]
        return adjacency

    def _weight(self, chemical, gap_junction):
        chemical = chemical.replace(np.nan, 0)
        gap_junction = gap_junction.replace(np.nan, 0)
        w_c_mask = torch.from_numpy(chemical.to_numpy(dtype=np.bool))
        w_g_mask = torch.from_numpy(gap_junction.to_numpy(dtype=np.bool))
        return w_c_mask, w_g_mask

    def _polarity_mask(self, synapses):
        """ chemical synaptic polarity mask
        synapses: [(pre1, post1), (pre2, post2), ...]
        excitatory mask: if True, the connection is excitatory
        inhibitory mask: if True, the connection is inhibitory
        """
        mask = pd.DataFrame(False, index=self.cells, columns=self.cells)
        for pre, post in synapses:
            mask.loc[pre, post] = True
        mask = torch.from_numpy(mask.to_numpy(dtype=np.bool))
        return mask

    def _external_input_mask(self, dim):
        """ external input bool mask
        dim: dimension of the external input
        flag: if True, sensory neurons receive external input; else, all neurons receive external input
        """
        mask = pd.DataFrame(False, index=list(range(dim)), columns=self.cells)
        mask.loc[:, self.sensory_neurons] = True
        mask = torch.from_numpy(mask.to_numpy(dtype=np.bool))
        return mask

    def _proprioception_mask(self):
        """ proprioception input synapse bool mask
        https://doi.org/10.1016/j.neuron.2012.08.039
        """
        mask = self._external_input_mask(dim=self.p)
        return mask

    def mask(self):
        """ mask generatation
        chemical mask
            mask_ij is True -> w_ij
            mask_ij is False -> w_ij = 0
        gap junction mask: symmetric
            mask_ij is True -> g_ij, g_ji
            mask_ij is False -> g_ij = g_ji = 0
        excitatory chemical mask: a sub mask of chemical mask
            mask_ij is True -> w_ij >= 0
            mask_ij is False -> w_ij
        inhibitory chemical mask: a sub mask of chemical mask, no overlap with excitatory chemical mask
            mask_ij is True -> w_ij <= 0
            mask_ij is False -> w_ij
        proprioception mask
            mask_pi is True -> w_pi
        """
        # chemical synapses and gap junctions
        w_c_mask, w_g_mask = self._weight(self.chemical, self.gap_junction)
        if torch.all(w_g_mask.tril() == w_g_mask.triu().T).item() is not True:
            raise AssertionError('Gap junction mask is not symmetric!')
        # chemical synapse polarity: subsets of chemical synapse mask, and no overlap between them
        w_c_ex_mask = self._polarity_mask(self.ex_synapses)
        w_c_in_mask = self._polarity_mask(self.in_synapses)
        w_c_ex_mask &= w_c_mask
        w_c_in_mask &= w_c_mask
        if torch.any(w_c_ex_mask & w_c_in_mask).item() is True:
            raise AssertionError('There is overlap between excitatory mask and inhibitory mask!')
        # chemical synapses exclude excitatory/inhibitory chemical synapses
        w_c_mask = w_c_mask ^ w_c_ex_mask ^ w_c_in_mask
        # combined chemical synapse polarity mask, (3, n, n)
        w_c_mask = torch.stack((w_c_mask, w_c_ex_mask, w_c_in_mask), dim=0)
        # proprioception input
        w_p_mask = self._proprioception_mask()
        # output
        muscles = set(self.muscles)
        output_index = torch.tensor([True if cell in muscles else False for cell in self.cells])
        return w_c_mask, w_g_mask, w_p_mask, output_index


def get_kwargs(path):
    """ connectome masks and params """
    muscles = body_wall_muscles()
    neurons = neuron_list2(path, muscles)
    sensory = sensory_neurons(path)
    print('{} neurons, {} muscles, {} sensory, {} cells in total'.format(len(neurons), len(muscles), len(sensory), len(neurons) + len(muscles)))
    p = 24
    connectome = Connectome(
        path=path, neurons=neurons, muscles=muscles, ex_synapses=[], in_synapses=[],
        sensory_neurons=sensory, p=p
    )
    w_c_mask, w_g_mask, w_p_mask, output_index = connectome.mask()
    return {
        'n': len(connectome.cells), 'm': len(connectome.muscles), 'p': p,
        'w_c_mask': w_c_mask, 'w_g_mask': w_g_mask, 'w_p_mask': w_p_mask, 'output_index': output_index
    }
