from matplotlib import pyplot as plt
import seaborn as sns
import torch
from virtual_nematode.connectomes.forward import neuron_list1, body_wall_muscles, chemical_synapse_polarity, neuron_list2
from virtual_nematode.networks.snn.forward import Connectome, DummyConnectome
from virtual_nematode.trainers.ncp import train_eval_test
import worm_assets


def plot_mask(w_c_mask, w_g_mask, w_c_ex_mask, w_c_in_mask, w_p_mask):
    plt.subplot(2, 3, 1)
    plt.title('chemical mask')
    sns.heatmap(w_c_mask, cmap='coolwarm', vmin=0, vmax=1)
    plt.subplot(2, 3, 2)
    plt.title('gap junction mask')
    sns.heatmap(w_g_mask, cmap='coolwarm', vmin=0, vmax=1)
    plt.subplot(2, 3, 3)
    plt.title('proprioception input mask')
    sns.heatmap(w_p_mask, cmap='coolwarm', vmin=0, vmax=1)
    plt.subplot(2, 3, 4)
    plt.title('ex chemical mask')
    sns.heatmap(w_c_ex_mask, cmap='coolwarm', vmin=0, vmax=1)
    plt.subplot(2, 3, 5)
    plt.title('in chemical mask')
    sns.heatmap(w_c_in_mask, cmap='coolwarm', vmin=0, vmax=1)
    plt.subplot(2, 3, 6)
    plt.show()


def train(model_name):
    if model_name == 'snn_forward':
        # torch.set_default_dtype(torch.float64)
        """ connectome: cells and synapse polarity """
        path = worm_assets.connectome_path(filename='SI 5 Connectome adjacency matrices, corrected July 2020.xlsx')
        muscles = body_wall_muscles()
        # neurons = neuron_list1()
        neurons = neuron_list2(path, muscles)
        ex_synapses, in_synapses = chemical_synapse_polarity()
        print('{} neurons, {} muscles, {} cells in total'.format(len(neurons), len(muscles), len(neurons) + len(muscles)))
        """ mask """
        p = 24
        connectome = Connectome(neurons, muscles, ex_synapses, in_synapses, path, p, p_mask=True, polarity_mask=False)
        # connectome = DummyConnectome(neurons, muscles, p, p_mask=True)
        w_c_mask, w_g_mask, w_c_ex_mask, w_c_in_mask, w_p_mask, output_index = connectome.mask()
        """ params """
        dt = 0.04
        n = len(connectome.cells)
        m = len(connectome.muscles)
        """ plot mask """
        # print(w_c_mask.dtype, w_g_mask.dtype, w_c_ex_mask.dtype, w_c_in_mask.dtype, w_p_mask.dtype, output_index.dtype)
        # plot_mask(w_c_mask, w_g_mask, w_c_ex_mask, w_c_in_mask, w_p_mask)
        """ trainer kwargs
        longer seq: data_name='data320.pt', lengths=[5000, 1000, 1000], batch_size=256, cuda=0, device_ids=[0, 1]
        """
        data_name = ['data_7000_5000_640_64_train.pt', 'data_7000_1000_640_64_eval.pt', 'data_7000_1000_640_64_test.pt']
        kwargs = {
            'data_name': data_name, 'model_name': model_name, 'batch_size': 256, 'seed': 11,
            'device_ids': [0, 1, 2, 3], 'lr': 0.01, 'epochs': 300, 'early_stop': 30, 'loss': 'MSELoss',
            # model kwargs
            'dt': dt, 'steps': 5, 'n': n, 'm': m, 'p': p, 'activation_type': 'sigmoid',
            'w_c_mask': w_c_mask, 'w_c_ex_mask': w_c_ex_mask, 'w_c_in_mask': w_c_in_mask,
            'w_g_mask': w_g_mask, 'w_p_mask': w_p_mask, 'output_index': output_index
        }
    elif model_name == 'ctrnn':
        # torch.set_default_dtype(torch.float64)
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
    train('snn_forward')
