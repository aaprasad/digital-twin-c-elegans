""" LIF neuron model """

import torch


class RateCell(torch.nn.Module):
    """ rate-coding LIF neuron model
    tau * dv/dt = -v + r * I
    I = x * W_I^T  + v * W_R^T
    reference: Long short-term memory and learning-to-learn in networks of spiking neurons
    """
    def __init__(self, tau, r):
        super(RateCell, self).__init__()
        self.tau = tau
        self.r = r  # resistance

    def forward(self, inputs, states):
        """ params
        inputs: I, external input
        states: v, membrane potential
        """
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
