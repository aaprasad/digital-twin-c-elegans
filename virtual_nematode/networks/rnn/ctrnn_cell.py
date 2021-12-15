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


class CTRNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, feedback=False, cell_clip=-1, unfolds=6, delta_t=0.1, tau=1):
        """ Continuous-time RNN
        input_size: input size
        hidden_size: amount of RNN hidden units
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
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feedback = feedback  # if True, use hidden state as part of input
        self.cell_clip = cell_clip
        self.unfolds = unfolds
        self.delta_t = delta_t
        self.tau = tau

    @property
    def state_size(self):
        return self.hidden_size

    def forward(self, inputs, states):
        inputs_prime = self.linear(inputs)  # if feedback is False
        for i in range(self.unfolds):
            if self.feedback is True:
                inputs_prime = self.linear(torch.cat((inputs, states), dim=-1))  # concat input and new hidden
            states_prime = (-states + inputs_prime) / self.tau  # dy/dt
            states = states + self.delta_t * states_prime
            if self.cell_clip > 0:
                states = torch.clamp(states, min=-self.cell_clip, max=self.cell_clip)
        outputs = states[:, 0:self.output_size]
        return outputs, states
