import torch
from virtual_nematode.connectomes.forward import neuron_list1, body_wall_muscles, chemical_synapse_polarity
from virtual_nematode.networks.snn.forward import Connectome, LinearConnectome
from virtual_nematode.trainers.ncp import train_eval_test
import worm_assets


def train(model_name):
    if model_name == 'snn_forward':
        torch.set_default_dtype(torch.float64)
        # connectome
        neurons = neuron_list1()
        muscles = body_wall_muscles()
        ex_synapses, in_synapses = chemical_synapse_polarity()
        path = worm_assets.connectome_path(filename='SI 5 Connectome adjacency matrices, corrected July 2020.xlsx')
        # connectome = Connectome(neurons, muscles, ex_synapses, in_synapses, path)
        connectome = LinearConnectome(neurons, muscles)
        # params
        dt = 0.04
        n = len(connectome.cells)
        p = 24
        mask_c, mask_g, ex_mask_c, in_mask_c, mask_output = connectome.mask()
        mask_p = connectome.proprioception_mask(p)
        """ trainer kwargs
        longer seq: data_name='data320.pt', lengths=[5000, 1000, 1000], batch_size=256, cuda=0, device_ids=[0, 1]
        """
        kwargs = {
            'data_name': 'data_640_64.pt', 'model_name': model_name, 'lengths': [50000, 10000, 10000], 'batch_size': 256, 'seed': 11,
            'cuda': 0, 'device_ids': [0, 1], 'lr': 0.001, 'epochs': 100, 'early_stop': 100, 'comment': '', 'loss': 'MSELoss',
            # model kwargs
            'dt': dt, 'steps': 5, 'n': n, 'p': p, 'activation_func': 'sigmoid',
            'mask_c': mask_c, 'ex_mask_c': ex_mask_c, 'in_mask_c': in_mask_c, 'mask_g': mask_g,
            'mask_p': mask_p, 'mask_output': mask_output
        }
    elif model_name == 'ctrnn':
        torch.set_default_dtype(torch.float64)
        kwargs = {
            'data_name': 'data32.pt', 'model_name': model_name, 'lengths': [50000, 10000, 10000], 'batch_size': 1024,
            'seed': 11, 'cuda': 0, 'device_ids': [0], 'lr': 0.001, 'weight_decay': 0, 'epochs': 100,
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
