import torch
from virtual_nematode.trainers.ncp import train_eval_test


def train(model_name):
    if model_name == 'fully_connected':
        """ results
        units = 100, batch_size = 1024
            memory: 7G * 4
            time: 2h / 300 epochs
        units = 72, batch_size = 1024
            memory: 4.9G * 4
            time: 2h / 300 epochs
        units = 150, batch_size = 512
            memory: 7.5G * 4
            time: 4h / 300 epochs
        """
        kwargs = {
            'data_name': 'ncp.pt', 'model_name': model_name, 'lengths': [48000, 12000, 12000],
            'batch_size': 1024, 'seed': 11, 'cuda': 0, 'device_ids': [0, 1, 2, 3], 'lr': 0.001, 'epochs': 100,
            'early_stop': 30, 'comment': '', 'loss': 'MSELoss',
            # model kwargs
            'units': 50, 'output_dim': 24, 'in_features': 48
        }
    elif model_name == 'ncp':
        """ results
        memory: 7G * 4
        time: 2h / 300 epochs
        """
        kwargs = {
            'data_name': 'ncp.pt', 'model_name': model_name, 'lengths': [48000, 12000, 12000], 'batch_size': 1024, 'seed': 11,
            'cuda': 0, 'device_ids': [0, 1, 2, 3], 'lr': 0.001, 'epochs': 100, 'early_stop': 30, 'comment': '', 'loss': 'MSELoss',
            # model kwargs
            'in_features': 48, 'inter_neurons': 24, 'command_neurons': 48, 'motor_neurons': 24, 'sensory_fanout': 24,
            'inter_fanout': 48, 'recurrent_command_synapses': 48, 'motor_fanin': 48
        }
    elif model_name == 'ctrnn':
        """ results
        hidden_size = 50/100/150
            memory: 0.8G
            time: 7min / 100 epochs
        """
        torch.set_default_dtype(torch.float64)
        kwargs = {
            'data_name': 'ncp.pt', 'model_name': model_name, 'lengths': [48000, 12000, 12000], 'batch_size': 1024, 'seed': 11,
            'cuda': 0, 'device_ids': [0], 'lr': 0.001, 'epochs': 100, 'early_stop': 30, 'comment': '', 'loss': 'MSELoss',
            # model kwargs
            'input_size': 48, 'hidden_size': 50, 'output_size': 24, 'feedback': True, 'readout': 'identity'
        }
    elif model_name == 'rnn':
        torch.set_default_dtype(torch.float64)
        kwargs = {
            'data_name': 'ncp.pt', 'model_name': model_name, 'lengths': [48000, 12000, 12000], 'batch_size': 1024, 'seed': 11,
            'cuda': 0, 'device_ids': [0], 'lr': 0.001, 'epochs': 100, 'early_stop': 30, 'comment': '', 'loss': 'MSELoss',
            # model kwargs
            'input_size': 48, 'hidden_size': 50, 'output_size': 24
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    train_eval_test(**kwargs)


if __name__ == '__main__':
    train(model_name='rnn')
