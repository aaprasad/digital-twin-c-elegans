import torch


class RNNCell2Stage(torch.nn.Module):
    def __init__(self, rnn_cell1, rnn_cell2, input1_range, input2_range):
        super(RNNCell2Stage, self).__init__()
        self.rnn_cell1 = rnn_cell1
        self.rnn_cell2 = rnn_cell2
        self.input1_range = input1_range  # tuple, (start, end)
        self.input2_range = input2_range  # tuple, (start, end)

    @property
    def state_size(self):
        return self.rnn_cell1.state_size + self.rnn_cell2.state_size

    def forward(self, inputs, states):
        inputs1 = inputs[:, self.input1_range[0]:self.input1_range[1]]  # rnn_cell1 inputs
        inputs2 = inputs[:, self.input2_range[0]:self.input2_range[1]]  # rnn_cell2 inputs
        states1 = states[:, 0:self.rnn_cell1.state_size]  # rnn_cell1 states
        states2 = states[:, self.rnn_cell1.state_size:]  # rnn_cell2 states
        outputs1, states1 = self.rnn_cell1(inputs1, states1)
        outputs2, states2 = self.rnn_cell2(inputs2 + outputs1, states2)
        states = torch.cat((states1, states2), dim=-1)
        return outputs2, states
