from virtual_nematode.connectomes.forward import Connectome as _Connectome


class Connectome(_Connectome):
    def __init__(self, gradient_size, gradient_mask, **kwargs):
        super(Connectome, self).__init__(**kwargs)
        self.gradient_size = gradient_size
        self.gradient_mask = gradient_mask

    def _gradient_mask(self):
        mask = self._external_input_mask(dim=self.gradient_size, flag=self.gradient_mask)
        return mask

    def mask(self):
        masks = super(Connectome, self).mask()
        w_gradient_mask = self._gradient_mask()
        return masks, w_gradient_mask
