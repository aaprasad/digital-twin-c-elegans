import numpy as np
import pandas as pd
import torch
from virtual_nematode.connectomes.cells import body_wall_muscles, cell_list
from virtual_nematode.connectomes.connections import (
    chemical_polarities,
    proprioception_connections,
    proprioception_connections1,
    proprioception_connections2
)


class Connectome(object):
    def __init__(self, path, cells, muscles, ex_synapses, in_synapses):
        self.path = path
        self.cells = cells
        self.muscles = muscles
        self.chemical, self.gap_junction = self._init()
        self.ex_synapses = self._check_synapses(ex_synapses)
        self.in_synapses = self._check_synapses(in_synapses)

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

    def _check_synapses(self, synapses):
        """ remove duplicate synapses, and check if related cells in the synapses exist """
        synapses = list(set(synapses))
        synapse_cells = set([pre for pre, _ in synapses] + [post for _, post in synapses])
        assert synapse_cells.issubset(set(self.cells))
        return synapses

    @staticmethod
    def _weight(adjacency):
        adjacency = adjacency.replace(np.nan, 0)
        mask = torch.from_numpy(adjacency.to_numpy(dtype=np.bool))
        return mask

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
        c_mask = self._weight(self.chemical)
        g_mask = self._weight(self.gap_junction)
        if torch.all(g_mask.tril() == g_mask.triu().T).item() is not True:
            raise AssertionError('Gap junction mask is not symmetric!')
        # chemical synapse polarity: subsets of chemical synapse mask, and no overlap between them
        c_ex_mask = self._polarity_mask(self.ex_synapses)
        c_in_mask = self._polarity_mask(self.in_synapses)
        c_ex_mask &= c_mask
        c_in_mask &= c_mask
        if torch.any(c_ex_mask & c_in_mask).item() is True:
            raise AssertionError('There is overlap between excitatory mask and inhibitory mask!')
        # chemical synapses exclude excitatory/inhibitory chemical synapses
        c_mask = c_mask ^ c_ex_mask ^ c_in_mask
        # combined chemical synapse polarity mask, (3, n, n)
        c_mask = torch.stack((c_mask, c_ex_mask, c_in_mask), dim=0)
        # output
        muscles = set(self.muscles)
        output_index = torch.tensor([True if cell in muscles else False for cell in self.cells])
        return c_mask, g_mask, output_index


class ExternalInput(object):
    def __init__(self, cells, dim, synapses, ex_synapses, in_synapses):
        self.cells = cells
        self.dim = dim  # input dimension
        self.synapses = self._check_synapses(synapses)
        self.ex_synapses = self._check_synapses(ex_synapses)
        self.in_synapses = self._check_synapses(in_synapses)

    def _check_synapses(self, synapses):
        """ check if related cells in the synapses exist """
        pre_cells = set([pre for pre, _ in synapses])
        assert pre_cells.issubset(set(range(self.dim)))
        post_cells = set([post for _, post in synapses])
        assert post_cells.issubset(set(self.cells))
        return synapses

    def _mask(self, synapses):
        """ polarity mask
        synapses: [(pre1, post1), (pre2, post2), ...]
        mask: if True, the connection is ex/in
        excitatory mask: if True, the connection is excitatory
        inhibitory mask: if True, the connection is inhibitory
        """
        mask = pd.DataFrame(False, index=list(range(self.dim)), columns=self.cells)
        for pre, post in synapses:
            mask.loc[pre, post] = True
        mask = torch.from_numpy(mask.to_numpy(dtype=np.bool))
        return mask

    def mask(self):
        mask = self._mask(self.synapses)
        ex_mask = self._mask(self.ex_synapses)
        in_mask = self._mask(self.in_synapses)
        ex_mask &= mask
        in_mask &= mask
        if torch.any(ex_mask & in_mask).item() is True:
            raise AssertionError('There is overlap between excitatory mask and inhibitory mask!')
        mask = mask ^ ex_mask ^ in_mask
        mask = torch.stack((mask, ex_mask, in_mask), dim=0)  # (3, dim, n)
        return mask


def get_kwargs(path, polarity_path):
    """ connectome masks and params """
    # connectome
    cells = cell_list(path)
    muscles = body_wall_muscles()
    print('{} cells, {} muscles'.format(len(cells), len(muscles)))
    ex_synapses, in_synapses = chemical_polarities(polarity_path)
    print('{} excitatory, {} inhibitory synapses'.format(len(ex_synapses), len(in_synapses)))
    connectome = Connectome(path=path, cells=cells, muscles=muscles, ex_synapses=ex_synapses, in_synapses=in_synapses)
    w_c_mask, w_g_mask, output_index = connectome.mask()
    # proprioception input
    p = 24
    # p_synapses, p_ex_synapses, p_in_synapses = proprioception_connections(dim=p)
    p_synapses, p_ex_synapses, p_in_synapses = proprioception_connections1(path=path, dim_muhead=7)
    # p_synapses, p_ex_synapses, p_in_synapses = proprioception_connections2(path=path, dim=p)
    print('{} total, {} excitatory, {} inhibitory proprioception synapses'.format(len(p_synapses), len(p_ex_synapses), len(p_in_synapses)))
    proprioception = ExternalInput(cells=cells, dim=p, synapses=p_synapses, ex_synapses=p_ex_synapses, in_synapses=p_in_synapses)
    w_p_mask = proprioception.mask()
    return {
        'n': len(connectome.cells), 'm': len(connectome.muscles), 'p': p,
        'w_c_mask': w_c_mask, 'w_g_mask': w_g_mask, 'w_p_mask': w_p_mask, 'output_index': output_index
    }
