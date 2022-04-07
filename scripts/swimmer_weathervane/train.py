""" swimmer: chemotaxis based on weathervane mechanism """

import torch
from virtual_nematode.trainers.ncp import train_eval_test


def train(model_name, data_name):
    if model_name == 'fully_connected':
        kwargs = {
            'data_name': data_name, 'model_name': model_name, 'lengths': [48000, 12000, 12000],
            'batch_size': 1024, 'seed': 11, 'cuda': 0, 'device_ids': [0, 1, 2, 3], 'lr': 0.001, 'epochs': 100,
            'early_stop': 30, 'comment': '', 'loss': 'MSELoss',
            # model kwargs
            'units': 100, 'output_dim': 24, 'in_features': 50
        }
    elif model_name == 'ctrnn':
        torch.set_default_dtype(torch.float64)
        kwargs = {
            'data_name': data_name, 'model_name': model_name, 'lengths': [48000, 12000, 12000], 'batch_size': 1024, 'seed': 11,
            'cuda': 0, 'device_ids': [0], 'lr': 0.001, 'epochs': 100, 'early_stop': 30, 'comment': '', 'loss': 'MSELoss',
            # model kwargs
            'input_size': 50, 'hidden_size': 100, 'output_size': 24, 'feedback': True, 'readout': 'identity'
        }
    elif model_name == 'ctrnn_2_stage':
        torch.set_default_dtype(torch.float64)
        kwargs = {
            'data_name': data_name, 'model_name': model_name, 'lengths': [48000, 12000, 12000], 'batch_size': 1024, 'seed': 11,
            'cuda': 0, 'device_ids': [0], 'lr': 0.001, 'epochs': 100, 'early_stop': 30, 'comment': '', 'loss': 'MSELoss',
            # model kwargs
            'input_size': 2, 'hidden_size': 50, 'output_size': 48, 'feedback': True, 'readout': 'identity',
            'load_kwargs': {'input_size': 48, 'hidden_size': 50, 'output_size': 24, 'feedback': True, 'readout': 'identity'},
            'input1_range': (48, 50), 'input2_range': (0, 48),
            'load_path': '/home/imc/disk1/virtual-nematode/scripts/swimmer_forward/runs/Dec16_16-02-23_h-10-176-50-34/model47.pt',
            'freeze': True
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    train_eval_test(**kwargs)


if __name__ == '__main__':
    """ experiments
    model_name: 'fully_connected', 'ctrnn', 'ctrnn_2_stage'
    data_name: 'steering.pt', 'steering_and_roaming.pt', 'full.pt'
    """
    train(model_name='ctrnn_2_stage', data_name='steering.pt')
