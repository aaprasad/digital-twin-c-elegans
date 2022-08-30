import torch
from virtual_nematode.connectomes.cells import cell_list
from virtual_nematode.connectomes.forward import get_kwargs as _get_kwargs


class SensoryInput(object):
    def __init__(self, cells, sensory):
        self.cells = cells
        self.sensory = self._check(sensory)

    def _check(self, cells):
        """ check if the cells exist """
        assert set(cells).issubset(set(self.cells))
        return cells

    def mask(self):
        mask = torch.tensor([self.cells.index(cell) for cell in self.sensory])
        return mask


def get_kwargs(path, polarity_path):
    """ connectome masks and params """
    cells = cell_list(path)
    sensory = ['ASEL', 'ASER']
    w_s_mask = SensoryInput(cells=cells, sensory=sensory)
    return {
        's': len(sensory), 'w_s_mask': w_s_mask,
        **_get_kwargs(path=path, polarity_path=polarity_path)
    }
