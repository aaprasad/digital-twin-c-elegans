import torch


class RNNCell(torch.nn.RNNCell):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        """ Vanilla RNN
        **kwargs: bias=True, nonlinearity='tanh', device=None
        """
        super(RNNCell, self).__init__(input_size, hidden_size, dtype=torch.float64, **kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size

    @property
    def state_size(self):
        return self.hidden_size

    def forward(self, inputs, states):
        states = super(RNNCell, self).forward(inputs, states)
        outputs = states[:, 0:self.output_size]
        return outputs, states
