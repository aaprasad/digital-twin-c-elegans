""" Continuous-time RNN
ODE: tau * dy/dt = -y + f(x * A^T + b)
Euler method:
    y(t + delta_t) = y(t) + delta_t * dy/dt
    dy/dt = (-y + f(x * A^T + b)) / tau
x: inputs or torch.cat((inputs, states), dim=-1) as input
    inputs: (input_size, )
    inputs + states: (input_size + hidden_size, )
A^T: transposed weight of linear layer
    inputs: (input_size, hidden_size)
    inputs + states: (input_size + hidden_size, hidden_size)
b: bias of linear layer, (hidden_size, )
f(): activation of linear layer
y: RNN hidden state
tau: time-constant of RNN cell
t: time step
https://github.com/mlech26l/keras-ncp/tree/master/reproducibility
"""

import torch
from virtual_nematode.networks.rnn.affine_activation import AffineActivation
from virtual_nematode.networks.rnn.identity import Identity


class CTRNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, feedback=True, readout='identity', cell_clip=-1, unfolds=6, delta_t=0.1, tau=1):
        """ Continuous-time RNN
        input_size: input size
        hidden_size: amount of RNN hidden units
        output_size: output size
        feedback: if True, use input and hidden concat as input
        readout: type of readout mapping, ['identity', 'affine', 'fully_connected']
        cell_clip: if > 0, clamp cell state
        unfolds: number of ODE solver steps
        delta_t: time of each ODE solver step
        tau: time-constant of the cell
        """
        super(CTRNNCell, self).__init__()
        if feedback is True:
            input_size = input_size + hidden_size
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Tanh()
        )
        if readout == 'identity':
            self.readout = Identity(output_size)
        elif readout == 'affine':
            self.readout = torch.nn.Sequential(
                Identity(output_size),
                AffineActivation(output_size)
            )
        elif readout == 'fully_connected':
            self.readout = torch.nn.Linear(hidden_size, output_size, bias=True)
        else:
            assert ValueError('Invalid readout type {}'.format(readout))
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feedback = feedback
        self.cell_clip = cell_clip
        self.unfolds = unfolds
        self.delta_t = delta_t
        self.tau = tau

    @property
    def state_size(self):
        return self.hidden_size

    def forward(self, inputs, states):
        if self.feedback is False:
            inputs_prime = self.linear(inputs)
        for i in range(self.unfolds):
            if self.feedback is True:
                inputs_prime = self.linear(torch.cat((inputs, states), dim=-1))  # concat input and new hidden
            states_prime = (-states + inputs_prime) / self.tau  # dy/dt
            states = states + self.delta_t * states_prime
            if self.cell_clip > 0:
                states = torch.clamp(states, min=-self.cell_clip, max=self.cell_clip)
        outputs = self.readout(states)
        return outputs, states
