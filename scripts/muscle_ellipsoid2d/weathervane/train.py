from virtual_nematode.connectomes.forward import body_wall_muscles, neuron_list2, sensory_neurons
from virtual_nematode.networks.snn.weathervane import Connectome
from virtual_nematode.trainers.ncp import train_eval_test
import worm_assets


def train(model_name):
    if model_name in ['snn_weathervane', 'snn2_weathervane']:
        """ connectome """
        path = worm_assets.connectome_path(filename='SI 5 Connectome adjacency matrices, corrected July 2020.xlsx')
        muscles = body_wall_muscles()
        neurons = neuron_list2(path, muscles)
        sensory = sensory_neurons(path)
        print('{} neurons, {} muscles, {} sensory, {} cells in total'.format(len(neurons), len(muscles), len(sensory), len(neurons) + len(muscles)))
        p = 24
        gradient_size = 1
        connectome = Connectome(
            gradient_size, gradient_mask=True,
            path=path, ex_synapses=[], in_synapses=[], polarity_mask=False,
            neurons=neurons, muscles=muscles, sensory_neurons=sensory, p=p, p_mask=True
        )
        (w_c_mask, w_g_mask, w_c_ex_mask, w_c_in_mask, w_p_mask, output_index), w_gradient_mask = connectome.mask()
        """ params """
        dt = 0.04
        n = len(connectome.cells)
        m = len(connectome.muscles)
        """ trainer kwargs """
        data_name = ['data_7000_5000_640_64_train.pt', 'data_7000_1000_640_64_eval.pt', 'data_7000_1000_640_64_test.pt']
        device_ids, batch_size, lr = [0, 1, 2, 3], 128, 5e-2
        strict, model_path = True, None
        # strict, model_path = False, '/home/imc/disk1/virtual-nematode/scripts/muscle_ellipsoid2d/forward/runs/Jul25_12-07-58_h-10-176-50-34/model119.pt'
        kwargs = {
            'data_name': data_name, 'model_name': model_name, 'batch_size': batch_size, 'seed': 11,
            'device_ids': device_ids, 'lr': lr, 'epochs': 300, 'early_stop': 300, 'loss': 'MSELoss',
            # model kwargs
            'dt': dt, 'steps': 5, 'n': n, 'm': m, 'p': p,
            'w_c_mask': w_c_mask, 'w_g_mask': w_g_mask, 'w_p_mask': w_p_mask, 'output_index': output_index,
            # 'w_c_ex_mask': w_c_ex_mask, 'w_c_in_mask': w_c_in_mask,
            'gradient_size': gradient_size, 'w_gradient_mask': w_gradient_mask,
            'model_path': model_path, 'strict': strict
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    train_eval_test(**kwargs)


if __name__ == '__main__':
    train('snn2_weathervane')
