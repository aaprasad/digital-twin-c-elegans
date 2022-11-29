from virtual_nematode.connectomes.forward import get_kwargs
from virtual_nematode.trainers.ncp import train_eval_test
import worm_assets


def train(model_name):
    if model_name.startswith('snn_forward') or model_name.startswith('li'):
        # data_name = ['data_7000_5000_640_64_train.pt', 'data_7000_1000_640_64_eval.pt', 'data_7000_1000_640_64_test.pt']
        data_name = ['data_7000_5000_640_64_stride8_n10_train.pt', 'data_7000_1000_640_64_stride8_n10_eval.pt', 'data_7000_1000_640_64_stride8_n10_test.pt']
        # data_name = ['data_7000_5000_640_64_stride4_n10_train.pt', 'data_7000_1000_640_64_stride4_n10_eval.pt', 'data_7000_1000_640_64_stride4_n10_test.pt']
        device_ids, batch_size, lr, weight_decay, epochs = [0, 1, 2, 3], 128, 5e-2, 0, 1000
        kwargs = {
            'data_name': data_name, 'model_name': model_name, 'batch_size': batch_size, 'seed': 42,  # 11, 42
            'device_ids': device_ids, 'lr': lr, 'weight_decay': weight_decay, 'epochs': epochs, 'early_stop': epochs,
            # 'model_path': None, 'strict': True, 'optimizer_path': None,
            'dt': 0.04, 'steps': 5, **get_kwargs(
                path=worm_assets.connectome_path(),
                polarity_path=worm_assets.polarity_path('Cook et al connectome.xls')  # 'NT+R method prediction.xls'
            )
        }
    elif model_name == 'ctrnn':
        kwargs = {
            'data_name': 'data32.pt', 'model_name': model_name, 'lengths': [50000, 10000, 10000], 'batch_size': 1024,
            'seed': 11, 'device_ids': [0], 'lr': 0.001, 'weight_decay': 0, 'epochs': 100,
            'early_stop': 100, 'comment': '', 'loss': 'MSELoss', 'sr': None,
            # model kwargs
            'input_size': 24, 'hidden_size': 171, 'output_size': 95, 'feedback': True, 'readout': 'identity',
            'unfolds': 6, 'delta_t': 0.1, 'tau': 1
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    print(kwargs)
    train_eval_test(**kwargs)


if __name__ == '__main__':
    # train('snn_forward3')
    # train('li_current')
    train('li_conductance')
    # train('li_conductance_cf')
