from virtual_nematode.connectomes.weathervane import get_kwargs
from virtual_nematode.trainers.ncp import train_eval_test
import worm_assets


def train(model_name):
    if model_name.startswith('snn_weathervane') or model_name.startswith('li'):
        # data_name = ['data_7000_5000_640_64_train.pt', 'data_7000_1000_640_64_eval.pt', 'data_7000_1000_640_64_test.pt']
        # data_name = ['data_7000_5000_640_64_stride4_n10_train.pt', 'data_7000_1000_640_64_stride4_n10_eval.pt', 'data_7000_1000_640_64_stride4_n10_test.pt']
        data_name = ['data_7000_5000_640_64_stride8_n10_train.pt', 'data_7000_1000_640_64_stride8_n10_eval.pt', 'data_7000_1000_640_64_stride8_n10_test.pt']
        # data_name = ['data_7000_5000_640_64_stride8_50000_train.pt', 'data_7000_1000_640_64_stride8_10000_eval.pt', 'data_7000_1000_640_64_stride8_10000_test.pt']
        # data_name = ['data_7000_5000_320_64_stride8_50000_train.pt', 'data_7000_1000_320_64_stride8_10000_eval.pt', 'data_7000_1000_320_64_stride8_10000_test.pt']
        # data_name = ['data_7000_640_256_stride8_n10_train.pt', 'data_7000_640_256_stride8_n10_eval.pt', 'data_7000_640_256_stride8_n10_test.pt']
        # data_name = ['data_7000_640_512_stride8_n10_train.pt', 'data_7000_640_512_stride8_n10_eval.pt', 'data_7000_640_512_stride8_n10_test.pt']
        # device_ids, batch_size, lr, weight_decay, epochs = [0, 1, 2, 3], 128, 5e-2, 0, 1000
        # device_ids, batch_size, lr, weight_decay, epochs = [0, 1], 128, 5e-2, 0, 1000
        # device_ids, batch_size, lr, weight_decay, epochs = [0, 1, 2, 3, 4, 5, 6, 7], 640, 0.05, 0, 1000
        device_ids, batch_size, lr, weight_decay, epochs = [0, 1, 2, 3], 288, 0.05, 0, 1000  # steps=5
        # device_ids, batch_size, lr, weight_decay, epochs = [4, 5, 6, 7], 288, 0.05, 0, 1000  # steps=5
        # device_ids, batch_size, lr, weight_decay, epochs = [0, 1, 2, 3], 160, 0.05, 0, 1000  # steps=10
        # device_ids, batch_size, lr, weight_decay, epochs = [4, 5, 6, 7], 160, 0.05, 0, 1000  # steps=10
        # device_ids, batch_size, lr, weight_decay, epochs = [0, 1, 2, 3], 72, 0.05, 0, 1000  # steps=20
        # device_ids, batch_size, lr, weight_decay, epochs = [4, 5, 6, 7], 72, 0.05, 0, 1000  # steps=20
        # device_ids, batch_size, lr, weight_decay, epochs = [0, 1, 2, 3], 288, 0.01, 0, 1000  # steps=5
        # device_ids, batch_size, lr, weight_decay, epochs = [4, 5, 6, 7], 288, 0.01, 0, 1000  # steps=5
        suffix, suffix_params = None, None
        # suffix, suffix_params = ['e_c'], {'lr': 5e-4}
        kwargs = {
            'data_name': data_name, 'model_name': model_name, 'batch_size': batch_size, 'seed': 48,
            'device_ids': device_ids, 'lr': lr, 'weight_decay': weight_decay, 'epochs': epochs, 'early_stop': epochs,
            'suffix': suffix, 'suffix_params': suffix_params,
            # model kwargs
            # 'model_path': None, 'strict': True, 'optimizer_path': None,
            'dt': 0.04, 'steps': 5,
            **get_kwargs(
                path=worm_assets.connectome_path(),
                polarity_path=worm_assets.polarity_path('Cook et al connectome.xls')
            ),
            # 'w_max': 1.
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    print(kwargs)
    train_eval_test(**kwargs)


if __name__ == '__main__':
    # train('snn_weathervane3')
    # train('li_conductance_gradient1')
    # train('li_conductance_gradient2')
    # train('li_conductance_gradient3')
    # train('li_current_gradient')
    # train('li_current_gradient1')
    # train('li_conductance_mixed_gradient')
    # train('li_conductance_mixed_gradient1')
    # train('li_conductance_mixed_gradient2')
    # train('li_conductance_gradient4')
    # train('li_conductance_unrestrained_gradient')
    # train('li_conductance_unrestrained_gradient1')
    # train('li_conductance_unrestrained_gradient2')
    # train('li_conductance_restrained_gradient')  # + '', '1', '2'
    # train('lig0')  # 'lig0', 'lig1', 'lig2'
    # train('lig1')
    # train('lig2')
    # train('lic')
    # train('lic51')
    train('lic41')
