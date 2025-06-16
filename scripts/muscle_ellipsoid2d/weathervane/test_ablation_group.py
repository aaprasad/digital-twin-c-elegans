""" command
python test_ablation_group.py Mar27_00-28-51_yulab-3090 model436.pt 0
"""

from analysis import get_results_numpy
import copy
import gymnasium as gym
import numpy as np
import os
from sim import position_func
import sys
from test import select_model
from test_vector_env import data_func, x_func, y_func
import torch
from virtual_nematode.connectomes.cells import cell_list
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer_weathervane_xml, make_swimmer_weathervane_fixed_fn
from virtual_nematode.testers.tester import test_vector_env_func
import worm_assets


def get_neuron_group():
    path = worm_assets.file_path(folder='connectomes', filename='neuron_group.txt')
    with open(path, 'r') as f:
        data = f.readlines()
    data = [d.strip() for d in data]
    data = [d.split(' ') for d in data]
    neuron_group = {}
    for i in range(0, len(data), 2):
        assert len(data[i]) == 1
        assert len(data[i + 1]) > 0
        neuron_group[data[i][0]] = data[i + 1]
    # check if neurons exist
    """
    cells = cell_list(path=worm_assets.connectome_path())
    print(len(cells), cells[0])
    cells = set(cells)
    for group, neurons in neuron_group.items():
        neurons_nonexist = []
        for n in neurons:
            if n not in cells:
                neurons_nonexist.append(n)
        if len(neurons_nonexist) > 0:
            print(group, neurons, neurons_nonexist)
    """
    # result
    print(len(neuron_group.keys()))
    return neuron_group


def modify_all_mask(model, ablation_type):
    ablation_cell_list = neuron_group[ablation_type]
    ablation_index_list = [cell_to_index[name] for name in ablation_cell_list]
    print('modify all mask: index {}, ablation index {}, ablation name {}'.format(type_to_index[ablation_type], ablation_index_list, ablation_cell_list))
    for index in ablation_index_list:
        model.cell.w_c_mask[:, index] = False
        model.cell.w_c_mask[index, :] = False
        model.cell.w_g_mask[:, index] = False
        model.cell.w_g_mask[index, :] = False
    return model


def test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type, suffix='', device_id=None, type_list=None):
    print(ckpt_name)
    print('ablation', ablation_type, device_id)
    model = select_model(model_folder, model_name, ckpt_name)
    device = torch.device('cuda:{}'.format(device_id) if device_id is not None and torch.cuda.is_available() else 'cpu')
    save_folder_temp = os.path.join(save_folder, ablation_type + suffix)
    os.makedirs(save_folder_temp, exist_ok=True)
    for tp in type_list:
        file_temp = os.path.join(save_folder_temp, 'test100.{}.{}.npz'.format(type_to_index[tp], tp))
        if os.path.exists(file_temp):
            continue
        model_temp = copy.deepcopy(model)
        if ablation_type == 'all':
            model_temp = modify_all_mask(model_temp, ablation_type=tp)
        else:
            raise AssertionError
        model_temp = model_temp.to(device)
        x, y = test_vector_env_func(env, model_temp, data_func, x_func, y_func, device, seed)
        print(x.shape, y.shape)
        np.savez(file_temp, x=x, y=y)
        get_results_numpy(x, y, max_episode_steps=max_episode_steps, sigma=5)


def main_fixed_4(pos=(10, 0), device_id=0):
    """ env """
    env = gym.vector.AsyncVectorEnv(
        env_fns=[
            make_swimmer_weathervane_fixed_fn(
                xml_str, reset_noise_scale=0.6, pos=pos, max_episode_steps=max_episode_steps,
                source=(0, 0), position_func=position_func
            ) for _ in range(num_envs)
        ]
    )
    print(env.action_space, env.observation_space)
    """ testing """
    distance = int(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
    suffix = '_d{}_group'.format(distance)

    if device_id in [0, 4]:
        type_list = types[0:24]
    elif device_id in [1, 5]:
        type_list = types[24:48]
    elif device_id in [2, 6]:
        type_list = types[48:72]
    elif device_id in [3, 7]:
        type_list = types[72:96]
    else:
        raise AssertionError
    test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', suffix=suffix, device_id=device_id, type_list=type_list)


if __name__ == '__main__':
    # load cell names
    cells = cell_list(path=worm_assets.connectome_path())
    cell_to_index = {n: i for i, n in enumerate(cells)}
    print(len(cells), cells[0])
    # settings
    runs_folder = sys.argv[1]
    ckpt_name = sys.argv[2]  # 'model.pt'
    device_id = int(sys.argv[3])  # 0~7
    model_folder = os.path.join('runs', runs_folder)
    save_folder = os.path.join('data', runs_folder, ckpt_name)
    video_folder = os.path.join('video', runs_folder, ckpt_name)
    assert os.path.exists(model_folder)
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)
    print(model_folder, ckpt_name)
    seed = 1024
    num_envs = 100
    max_episode_steps = 2500
    # env
    xml_str = make_swimmer_weathervane_xml(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        source=(0, 0), density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    # testing
    neuron_group = get_neuron_group()
    types = sorted(neuron_group.keys())
    type_to_index = {n: i for i, n in enumerate(types)}
    model_name = 'lic62'
    main_fixed_4(pos=(10, 0), device_id=device_id)
