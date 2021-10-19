import torch
import torch.nn as nn


class RNNSequence(nn.Module):
    def __init__(self, rnn_cell):
        super(RNNSequence, self).__init__()
        self.rnn_cell = rnn_cell

    def forward(self, x):
        """ unfolds a RNN cell into a sequence
        https://colab.research.google.com/drive/1VWoGcpyqGvrUOUzH7ccppE__m-n1cAiI?usp=sharing
        """
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device
        )
        outputs = []
        for t in range(seq_len):
            inputs = x[:, t]
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            outputs.append(new_output)
        outputs = torch.stack(outputs, dim=1)  # return entire sequence
        return outputs

    def step(self, x, hidden_state=None):
        """ run RNN cell step by step for online testing """
        if hidden_state is None:
            batch_size = x.size(0)
            hidden_state = torch.zeros((batch_size, self.rnn_cell.state_size))
        output, hidden_state = self.rnn_cell.forward(x, hidden_state)
        return output, hidden_state
