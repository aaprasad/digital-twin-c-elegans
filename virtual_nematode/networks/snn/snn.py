import torch


class SNN(torch.nn.Module):
    def __init__(self, cell):
        super(SNN, self).__init__()
        self.cell = cell

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        state = torch.zeros((batch_size, self.cell.state_size), device=device)
        output = []
        for t in range(seq_len):
            inputs = x[:, t]
            new_output, state = self.cell.forward(inputs, state)
            output.append(new_output)
        output = torch.stack(output, dim=1)  # return entire sequence
        return output

    def step(self, x, state=None):
        if state is None:
            batch_size = x.size(0)
            state = torch.zeros((batch_size, self.cell.state_size))
        output, state = self.cell.forward(x, state)
        return output, state
