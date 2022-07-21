from data import x_func as data_func
import gym
import os
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.connectomes.forward import body_wall_muscles, chemical_synapse_polarity, neuron_list2, sensory_neurons
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.networks.snn.forward import Connectome
from virtual_nematode.testers.forward import tester, single_tester, test_func1, test_func2
from virtual_nematode.trainers.ncp import prepare_model
import worm_assets


def y_func(action, **kwargs):
    return action.tolist()


def y_func1(state, **kwargs):
    return state.squeeze(dim=0).tolist()


def select_model(model_folder, model_name, ckpt_name):
    if model_name == 'snn_forward' or model_name == 'snn1_forward':
        """ connectome: cells and synapse polarity """
        path = worm_assets.connectome_path(filename='SI 5 Connectome adjacency matrices, corrected July 2020.xlsx')
        muscles = body_wall_muscles()
        neurons = neuron_list2(path, muscles)
        sensory = sensory_neurons(path)
        # ex_synapses, in_synapses = chemical_synapse_polarity()
        print('{} cells'.format(len(neurons) + len(muscles)), end=' ')
        """ mask """
        p = 24
        connectome = Connectome(
            path, ex_synapses=[], in_synapses=[], polarity_mask=False,
            neurons=neurons, muscles=muscles, sensory_neurons=sensory, p=p, p_mask=True
        )
        w_c_mask, w_g_mask, w_c_ex_mask, w_c_in_mask, w_p_mask, output_index = connectome.mask()
        """ params """
        dt = 0.04
        n = len(connectome.cells)
        m = len(connectome.muscles)
        """ eval kwargs """
        kwargs = {
            'dt': dt, 'steps': 5, 'n': n, 'm': m, 'p': p,
            'w_c_mask': w_c_mask, 'w_g_mask': w_g_mask, 'w_p_mask': w_p_mask, 'output_index': output_index,
            # 'w_c_ex_mask': w_c_ex_mask, 'w_c_in_mask': w_c_in_mask
        }
    elif model_name == 'ctrnn':
        kwargs = {
            'input_size': 24, 'hidden_size': 171, 'output_size': 95, 'feedback': True, 'readout': 'identity',
            'unfolds': 6, 'delta_t': 0.1, 'tau': 1
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    model = prepare_model(model_name, model_path=os.path.join(model_folder, ckpt_name), **kwargs)
    return model


def evaluate(model_folder, model_name, start, end):
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        print(ckpt_name, end=' ')
        model = select_model(model_folder, model_name, ckpt_name)
        if model_name == 'snn_forward' or model_name == 'snn1_forward':
            test_func = test_func2
        elif model_name == 'ctrnn':
            test_func = test_func1
        else:
            raise AssertionError('{} not exist'.format(model_name))
        _, y = single_tester(env, model, data_func, x_func, y_func, seed, max_episode_steps, test_func=test_func)
        torch.save(y, os.path.join(data_path, 'model{}.pt'.format(i)))  # action sequence
        # torch.save(y, os.path.join(data_path, 'model{}.state.pt'.format(i)))


def test(model_folder, model_name, start, end):
    """ online test multiple trials for testing """
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        print(ckpt_name, end=' ')
        model = select_model(model_folder, model_name, ckpt_name)
        if model_name == 'snn_forward' or model_name == 'snn1_forward':
            test_func = test_func2
        elif model_name == 'ctrnn':
            test_func = test_func1
        else:
            raise AssertionError('{} not exist'.format(model_name))
        tester(env, model, data_func, x_func, y_func, seed, max_episode_steps, model_folder, model_name, data_size=100, disable=True, test_func=test_func)


def record(model_folder, model_name, env, ckpt_name):
    """ online test once for evaluation and record video """
    print(ckpt_name, end=' ')
    model = select_model(model_folder, model_name, ckpt_name)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder, ckpt_name), force=True)
    if model_name == 'snn_forward' or model_name == 'snn1_forward':
        test_func = test_func2
    elif model_name == 'ctrnn':
        test_func = test_func1
    else:
        raise AssertionError('{} not exist'.format(model_name))
    _, y = single_tester(env, model, data_func, x_func, y_func, seed, max_episode_steps, test_func=test_func)
    torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    model_folder = os.path.join('runs', runs_folder)
    if os.path.exists(model_folder) is False:
        raise ValueError('Invalid runs folder {}'.format(runs_folder))
    seed = 42
    max_episode_steps = 2500
    data_path = os.path.join('data', runs_folder)
    os.makedirs(data_path, exist_ok=True)
    env = make_swimmer(
        n_bodies=25, joint_range='-90 90', max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        density=1.2, viscosity=0.1, condim=3, friction='1 1'
    )
    # evaluate(model_folder, 'snn1_forward', start=0, end=100)
    test(model_folder, 'snn1_forward', start=0, end=100)
    # record(model_folder, 'snn1_forward', env, ckpt_name='model.pt')
