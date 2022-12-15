from analysis import get_results_numpy
import copy
import gym
import numpy as np
import os
import sys
from test import select_model
from test_ablation_once import modify_all_mask, modify_chemical_mask, modify_electrical_mask
from test_vector_env import data_func, x_func, y_func
import torch
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer_xml, make_swimmer_fn
from virtual_nematode.testers.tester import test_vector_env_func


def test(model_folder, model_name, ckpt_name, save_folder, ablation_type, device_id=None, id_list=None):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    device = torch.device('cuda:{}'.format(device_id) if device_id is not None and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    save_folder_temp = os.path.join(save_folder, ablation_type)
    os.makedirs(save_folder_temp, exist_ok=True)
    for i in id_list:
        file_temp = os.path.join(save_folder_temp, 'test100.{}.npz'.format(i))
        if os.path.exists(file_temp):
            continue
        model_temp = copy.deepcopy(model)
        if ablation_type == 'all':
            model_temp = modify_all_mask(model_temp, index=i)
        elif ablation_type == 'chemical':
            model_temp = modify_chemical_mask(model_temp, index=i)
        elif ablation_type == 'electrical':
            model_temp = modify_electrical_mask(model_temp, index=i)
        else:
            raise AssertionError
        x, y = test_vector_env_func(env, model_temp, data_func, x_func, y_func, device, seed)
        print(x.shape, y.shape)
        np.savez(file_temp, x=x, y=y)
        get_results_numpy(x, y, max_episode_steps=max_episode_steps)


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    ckpt_name = sys.argv[2]  # 'model.pt'
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
    """ env """
    xml_str = make_swimmer_xml(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    env = gym.vector.AsyncVectorEnv(env_fns=[make_swimmer_fn(xml_str, max_episode_steps, reset_noise_scale=0.6) for _ in range(num_envs)])
    print(env.action_space, env.observation_space)
    """ testing """
    model_name = 'li_conductance'
    test(model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=0, id_list=list(range(0, 235)))
    test(model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=1, id_list=list(range(235, 469)))
    test(model_folder, model_name, ckpt_name, save_folder, ablation_type='chemical', device_id=2, id_list=list(range(0, 235)))
    test(model_folder, model_name, ckpt_name, save_folder, ablation_type='chemical', device_id=3, id_list=list(range(235, 469)))
    test(model_folder, model_name, ckpt_name, save_folder, ablation_type='electrical', device_id=4, id_list=list(range(0, 235)))
    test(model_folder, model_name, ckpt_name, save_folder, ablation_type='electrical', device_id=5, id_list=list(range(235, 469)))
