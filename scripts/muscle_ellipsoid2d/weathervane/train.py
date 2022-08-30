from virtual_nematode.connectomes.weathervane import get_kwargs
from virtual_nematode.trainers.ncp import train_eval_test
import worm_assets


def train(model_name):
    if model_name.startswith('snn_weathervane'):
        data_name = ['data_7000_5000_640_64_train.pt', 'data_7000_1000_640_64_eval.pt', 'data_7000_1000_640_64_test.pt']
        device_ids, batch_size, lr, epochs = [0, 1, 2, 3], 128, 5e-2, 1000
        # strict, model_path = True, None
        strict, model_path = False, '/home/imc/disk1/virtual-nematode/scripts/muscle_ellipsoid2d/forward/runs/Jul25_12-07-58_h-10-176-50-34/model119.pt'
        kwargs = {
            'data_name': data_name, 'model_name': model_name, 'batch_size': batch_size, 'seed': 11,
            'device_ids': device_ids, 'lr': lr, 'epochs': epochs, 'early_stop': epochs, 'loss': 'MSELoss',
            # model kwargs
            'model_path': model_path, 'strict': strict,
            'dt': 0.04, 'steps': 5,
            **get_kwargs(
                path=worm_assets.connectome_path(),
                polarity_path=worm_assets.polarity_path('Cook et al connectome.xls')
            )
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    train_eval_test(**kwargs)


if __name__ == '__main__':
    train('snn_weathervane3')
