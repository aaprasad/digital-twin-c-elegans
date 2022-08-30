import torch
from virtual_nematode.connectomes.cells import cell_list
from virtual_nematode.connectomes.forward import get_kwargs as _get_kwargs


class SensoryInput(object):
    def __init__(self, cells, sensory_cells):
        self.cells = cells
        self.sensory_cells = self._check(sensory_cells)

    def _check(self, cells):
        """ check if the cells exist """
        assert set(cells).issubset(set(self.cells))
        return cells

    def mask(self):
        mask = torch.tensor([self.cells.index(cell) for cell in self.sensory_cells])
        return mask


def get_kwargs(path, polarity_path):
    """ connectome masks and params """
    cells = cell_list(path)
    sensory_cells = ['ASEL', 'ASER']
    print('sensory cells', sensory_cells)
    sensation = SensoryInput(cells=cells, sensory_cells=sensory_cells)
    w_s_mask = sensation.mask()
    return {
        's': len(sensory_cells), 'w_s_mask': w_s_mask,
        **_get_kwargs(path=path, polarity_path=polarity_path)
    }
