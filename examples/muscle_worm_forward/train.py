from virtual_nematode.trainers.ncp import train_eval_test


def train(model_name):
    if model_name == 'fully_connected':
        """ results
        units = 200, batch_size = 256
            memory: 7.5G * 4
            time: 7h 30min / 300 epochs
        units = 150, batch_size = 512
            memory: 8.6G * 4
            time: 4h 30min / 300 epochs
        units = 100, batch_size = 1024
            memory: 8.5G * 4
            time: 2h 20min / 300 epochs
        units = 128, batch_size = 512
            memory: 6.7G * 4
            time: 3h 45min / 300 epochs
        """
        kwargs = {
            'data_name': 'ncp.pt', 'model_name': model_name, 'lengths': [48000, 12000, 12000], 'batch_size': 1024,
            'seed': 11, 'cuda': 0, 'device_ids': [0, 1, 2, 3], 'lr': 0.001, 'weight_decay': 0, 'epochs': 100,
            'early_stop': 30, 'comment': '', 'loss': 'MSELoss', 'sr': None,
            # model kwargs
            'units': 100, 'output_dim': 96, 'in_features': 192, 'output_mapping': 'affine'
        }
    elif model_name == 'ctrnn':
        kwargs = {
            'data_name': 'ncp.pt', 'model_name': model_name, 'lengths': [48000, 12000, 12000], 'batch_size': 1024,
            'seed': 11, 'cuda': 0, 'device_ids': [0], 'lr': 0.001, 'weight_decay': 0, 'epochs': 100,
            'early_stop': 30, 'comment': '', 'loss': 'MSELoss', 'sr': None,
            # model kwargs
            'input_size': 192, 'hidden_size': 100, 'output_size': 96, 'feedback': True, 'readout': 'affine'
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    train_eval_test(**kwargs)


if __name__ == '__main__':
    train(model_name='ctrnn')
