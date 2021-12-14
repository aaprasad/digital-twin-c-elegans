import torch


class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        """ Vanilla RNN
        **kwargs: bias=True, nonlinearity='tanh'
        """
        super(RNNCell, self).__init__()
        # self.rnn_cell = torch.nn.RNNCell(input_size, hidden_size, **kwargs)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Tanh()
        )
        self.hidden_size = hidden_size
        self.output_size = output_size

    @property
    def state_size(self):
        return self.hidden_size

    def forward(self, inputs, states):
        # states = self.rnn_cell(inputs, states)
        states = self.linear(torch.cat((inputs, states), dim=-1))
        outputs = states[:, 0:self.output_size]
        return outputs, states
