from virtual_nematode.connectomes.forward import get_kwargs
from virtual_nematode.trainers.ncp import train_eval_test


def train():
    data_name = ['data_7000_5000_640_64_stride8_n10_train.pt', 'data_7000_1000_640_64_stride8_n10_eval.pt', 'data_7000_1000_640_64_stride8_n10_test.pt']
    device_ids, batch_size, lr, weight_decay, epochs = [0, 1, 2, 3], 128, 5e-2, 0, 1000
    kwargs = {
        'data_name': data_name, 'model_name': 'li_conductance', 'batch_size': batch_size, 'seed': 42,
        'device_ids': device_ids, 'lr': lr, 'weight_decay': weight_decay, 'epochs': epochs, 'early_stop': epochs,
        'dt': 0.04, 'steps': 5, **get_kwargs(
            path='data/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx',
            polarity_path='data/Cook et al connectome.xls'
        )
    }
    print(kwargs)
    train_eval_test(**kwargs)


if __name__ == '__main__':
    train()
