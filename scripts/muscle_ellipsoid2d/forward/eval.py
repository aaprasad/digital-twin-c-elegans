from data import x_func as data_func
import gym
from matplotlib import pyplot as plt
import os
import seaborn as sns
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.connectomes.forward import neuron_list1, body_wall_muscles, chemical_synapse_polarity
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.networks.snn.forward import Connectome, LinearConnectome
from virtual_nematode.testers.forward import tester, single_tester, test_func1, test_func2
from virtual_nematode.trainers.ncp import prepare_model
import worm_assets


def show_weight(model):
    plt.figure(figsize=(10, 15))
    # chemical weight
    w_c = (
        model.cell.w_c.abs() * model.cell.ex_mask_c -
        model.cell.w_c.abs() * model.cell.in_mask_c +
        model.cell.w_c * (
            model.cell.mask_c.float() -
            model.cell.ex_mask_c.float() -
            model.cell.in_mask_c.float()
        )
    )
    w_c = w_c.clone().detach()
    w_c_max = w_c.abs().max().item()
    plt.subplot(2, 3, 1)
    plt.title('chemical weight')
    sns.heatmap(w_c, cmap='coolwarm', vmin=-w_c_max, vmax=w_c_max)
    # proprioception input weight
    w_p = model.cell.w_p * model.cell.mask_p
    w_p = w_p.clone().detach()
    w_p_max = w_p.abs().max().item()
    plt.subplot(2, 3, 2)
    plt.title('proprioception input weight')
    sns.heatmap(w_p, cmap='coolwarm', vmin=-w_p_max, vmax=w_p_max)
    # tau
    tau = model.cell.bias.clone().detach()
    plt.subplot(2, 3, 3)
    plt.title('tau')
    plt.bar(tau)
    # gap junction weight
    w_g = model.cell.w_g.abs()
    w_g = (w_g.tril() + w_g.tril(diagonal=-1).T) * model.cell.mask_g
    w_g = w_g.clone().detach()
    w_g_max = w_g.max().item()
    plt.subplot(2, 3, 4)
    plt.title('gap junction weight')
    sns.heatmap(w_g, cmap='coolwarm', vmin=0, vmax=w_g_max)
    # output scaling weight
    w_output = model.cell.w_output.clone().detach()
    plt.subplot(2, 3, 5)
    plt.title('output scaling weight')
    plt.bar(w_output)
    # bias
    bias = model.cell.bias.clone().detach()
    plt.subplot(2, 3, 6)
    plt.title('bias')
    plt.bar(bias)
    plt.savefig('network.png')


def select_model(model_name, ckpt_name):
    if model_name == 'snn_forward':
        torch.set_default_dtype(torch.float64)
        # connectome
        neurons = neuron_list1()
        muscles = body_wall_muscles()
        ex_synapses, in_synapses = chemical_synapse_polarity()
        path = worm_assets.connectome_path(filename='SI 5 Connectome adjacency matrices, corrected July 2020.xlsx')
        connectome = Connectome(neurons, muscles, ex_synapses, in_synapses, path)
        # connectome = LinearConnectome(neurons, muscles)
        # params
        dt = 0.04
        n = len(connectome.cells)
        m = len(connectome.muscles)
        p = 24
        mask_c, mask_g, ex_mask_c, in_mask_c, mask_output = connectome.mask(polarity_mask=True)
        mask_p = connectome.proprioception_mask(p, p_mask=True)
        kwargs = {
            'dt': dt, 'steps': 5, 'n': n, 'm': m, 'p': p, 'activation_type': 'sigmoid',
            'mask_c': mask_c, 'ex_mask_c': ex_mask_c, 'in_mask_c': in_mask_c, 'mask_g': mask_g,
            'mask_p': mask_p, 'mask_output': mask_output
        }
    elif model_name == 'ctrnn':
        torch.set_default_dtype(torch.float64)
        kwargs = {
            'input_size': 24, 'hidden_size': 171, 'output_size': 95, 'feedback': True, 'readout': 'identity',
            'unfolds': 6, 'delta_t': 0.1, 'tau': 1
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    model = prepare_model(model_name, model_path=os.path.join(model_folder, ckpt_name), **kwargs)
    # show_weight(model)
    # print(torch.all(mask_c == model.cell.mask_c))
    # print(torch.all(mask_g == model.cell.mask_g))
    # print(torch.all(ex_mask_c == model.cell.ex_mask_c))
    # print(torch.all(in_mask_c == model.cell.in_mask_c))
    # print(torch.all(mask_output == model.cell.mask_output))
    # print(torch.all(mask_p == model.cell.mask_p))
    return model


def evaluate(model_name, start, end):
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        model = select_model(model_name, ckpt_name)
        print(ckpt_name, end=' ')
        if model_name == 'snn_forward':
            test_func = test_func2
        elif model_name == 'ctrnn':
            test_func = test_func1
        else:
            raise AssertionError('{} not exist'.format(model_name))
        _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps, test_func=test_func)
        torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


def test(model_name, start, end):
    """ online test multiple trials for testing """
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        print(ckpt_name, end=' ')
        model = select_model(model_name, ckpt_name)
        if model_name == 'snn_forward':
            test_func = test_func2
        elif model_name == 'ctrnn':
            test_func = test_func1
        else:
            raise AssertionError('{} not exist'.format(model_name))
        tester(env, model, data_func, x_func, seed, max_episode_steps, model_folder, model_name, data_size=100, disable=True, test_func=test_func)


def record(model_name, env, ckpt_name):
    """ online test once for evaluation and record video """
    model = select_model(model_name, ckpt_name)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder, ckpt_name), force=True)
    if model_name == 'snn_forward':
        test_func = test_func2
    elif model_name == 'ctrnn':
        test_func = test_func1
    else:
        raise AssertionError('{} not exist'.format(model_name))
    _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps, test_func=test_func)
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
    evaluate('snn_forward', start=0, end=100)
    # test('snn_forward', start=0, end=100)
    # record('snn_forward', env, ckpt_name='model.pt')
