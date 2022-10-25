from virtual_nematode.connectomes.forward import get_kwargs
from virtual_nematode.trainers.ncp import train_eval_test
import worm_assets


def train(model_name):
    if model_name.startswith('snn_forward') or model_name.startswith('li'):
        data_name = ['data_7000_5000_640_64_train.pt', 'data_7000_1000_640_64_eval.pt', 'data_7000_1000_640_64_test.pt']
        # data_name = ['data_3500_2500_1280_64_train.pt', 'data_3500_500_1280_64_eval.pt', 'data_3500_500_1280_64_test.pt']
        device_ids, batch_size, lr, epochs = [0, 1, 2, 3], 128, 5e-2, 1000
        kwargs = {
            'data_name': data_name, 'model_name': model_name, 'batch_size': batch_size, 'seed': 11,
            'device_ids': device_ids, 'lr': lr, 'weight_decay': 0, 'epochs': epochs, 'early_stop': epochs,
            # 'model_path': '/home/imc/disk1/virtual-nematode/scripts/muscle_ellipsoid2d/forward/runs/Sep06_10-08-29_h-10-176-50-34/model999.pt',
            # 'strict': True,
            # 'optimizer_path': '/home/imc/disk1/virtual-nematode/scripts/muscle_ellipsoid2d/forward/runs/Sep06_10-08-29_h-10-176-50-34/model999.optim.pt',
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
    train_eval_test(**kwargs)


if __name__ == '__main__':
    # train('snn_forward3')
    # train('li_current')
    train('li_conductance')
