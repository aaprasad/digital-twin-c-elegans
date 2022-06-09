from data import x_func as data_func
import gym
import os
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.connectomes.forward import neuron_list1, body_wall_muscles, chemical_synapse_polarity
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.networks.snn.forward import Connectome
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model
import worm_assets


def select_model(ckpt_name):
    # connectome
    neurons = neuron_list1()
    muscles = body_wall_muscles()
    ex_synapses, in_synapses = chemical_synapse_polarity()
    path = worm_assets.connectome_path(filename='SI 5 Connectome adjacency matrices, corrected July 2020.xlsx')
    connectome = Connectome(neurons, muscles, ex_synapses, in_synapses, path)
    # params
    freq = 0.04
    n = len(connectome.cells)
    p = 24
    mask_c, mask_g, ex_mask_c, in_mask_c, mask_output = connectome.mask()
    mask_p = connectome.proprioception_mask(p)
    kwargs = {
        'freq': freq, 'n': n, 'p': p,
        'mask_c': mask_c, 'ex_mask_c': ex_mask_c, 'in_mask_c': in_mask_c, 'mask_g': mask_g,
        'mask_p': mask_p, 'mask_output': mask_output
    }
    model = prepare_model(model_name, model_path=os.path.join(model_folder, ckpt_name), **kwargs)
    # print(torch.all(mask_c == model.cell.mask_c))
    # print(torch.all(mask_g == model.cell.mask_g))
    # print(torch.all(ex_mask_c == model.cell.ex_mask_c))
    # print(torch.all(in_mask_c == model.cell.in_mask_c))
    # print(torch.all(mask_output == model.cell.mask_output))
    # print(torch.all(mask_p == model.cell.mask_p))
    return model


def evaluate(start, end):
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        model = select_model(ckpt_name)
        print(ckpt_name, end=' ')
        _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps)
        torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


def test(start, end):
    """ online test multiple trials for testing """
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        print(ckpt_name, end=' ')
        model = select_model(ckpt_name)
        tester(env, model, data_func, x_func, seed, max_episode_steps, model_folder, model_name, data_size=100, disable=True)


def record(env, ckpt_name):
    """ online test once for evaluation and record video """
    model = select_model(ckpt_name)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder, ckpt_name), force=True)
    _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps)
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
        n_bodies=25, joint_range='-90 90', max_episode_steps=max_episode_steps, reset_noise_scale=0.7,
        density=1.2, viscosity=0.1, condim=3, friction='1 1'
    )
    model_name = 'snn_forward'
    evaluate(start=0, end=100)
    # test(start=0, end=100)
    # record(env, ckpt_name='model.pt')
