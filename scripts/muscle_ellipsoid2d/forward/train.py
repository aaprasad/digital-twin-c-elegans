from matplotlib import pyplot as plt
import seaborn as sns
from virtual_nematode.connectomes.forward import body_wall_muscles, chemical_synapse_polarity, neuron_list2, sensory_neurons
from virtual_nematode.networks.snn.forward import Connectome
from virtual_nematode.trainers.ncp import train_eval_test
import worm_assets


def plot_mask(w_c_mask, w_g_mask, w_c_ex_mask, w_c_in_mask, w_p_mask):
    plt.subplot(2, 3, 1)
    plt.title('Chemical Synapse Mask')
    sns.heatmap(w_c_mask, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Cell ID')
    plt.subplot(2, 3, 2)
    plt.title('Gap Junction Mask')
    sns.heatmap(w_g_mask, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Cell ID')
    plt.subplot(2, 3, 3)
    plt.title('Proprioception Input Mask')
    sns.heatmap(w_p_mask, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Joint ID')
    # plt.subplot(2, 3, 4)
    # plt.title('ex chemical mask')
    # sns.heatmap(w_c_ex_mask, cmap='coolwarm', vmin=0, vmax=1)
    # plt.subplot(2, 3, 5)
    # plt.title('in chemical mask')
    # sns.heatmap(w_c_in_mask, cmap='coolwarm', vmin=0, vmax=1)
    # plt.subplot(2, 3, 6)
    plt.show()


def train(model_name):
    if model_name == 'snn_forward' or model_name == 'snn1_forward':
        """ connectome: cells and synapse polarity """
        path = worm_assets.connectome_path(filename='SI 5 Connectome adjacency matrices, corrected July 2020.xlsx')
        muscles = body_wall_muscles()
        neurons = neuron_list2(path, muscles)
        sensory = sensory_neurons(path)
        # ex_synapses, in_synapses = chemical_synapse_polarity()
        print('{} neurons, {} muscles, {} sensory, {} cells in total'.format(len(neurons), len(muscles), len(sensory), len(neurons) + len(muscles)))
        """ mask """
        p = 24
        connectome = Connectome(
            path, ex_synapses=[], in_synapses=[], polarity_mask=False,
            neurons=neurons, muscles=muscles, sensory_neurons=sensory, p=p, p_mask=True
        )
        w_c_mask, w_g_mask, w_c_ex_mask, w_c_in_mask, w_p_mask, output_index = connectome.mask()
        # plot_mask(w_c_mask, w_g_mask, w_c_ex_mask, w_c_in_mask, w_p_mask)
        """ params """
        dt = 0.04
        n = len(connectome.cells)
        m = len(connectome.muscles)
        """ trainer kwargs
        longer seq: data_name='data320.pt', lengths=[5000, 1000, 1000], batch_size=256, cuda=0, device_ids=[0, 1]
        """
        data_name = ['data_7000_5000_640_64_train.pt', 'data_7000_1000_640_64_eval.pt', 'data_7000_1000_640_64_test.pt']
        # device_ids, batch_size, lr = [0, 1], 256, 1e-2
        device_ids, batch_size, lr = [0, 1, 2, 3], 128, 1e-2
        kwargs = {
            'data_name': data_name, 'model_name': model_name, 'batch_size': batch_size, 'seed': 11,
            'device_ids': device_ids, 'lr': lr, 'epochs': 1000, 'early_stop': 1000, 'loss': 'MSELoss',
            # model kwargs
            'dt': dt, 'steps': 5, 'n': n, 'm': m, 'p': p,
            'w_c_mask': w_c_mask, 'w_g_mask': w_g_mask, 'w_p_mask': w_p_mask, 'output_index': output_index,
            # 'w_c_ex_mask': w_c_ex_mask, 'w_c_in_mask': w_c_in_mask
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
    train('snn1_forward')  # 'snn_forward'
