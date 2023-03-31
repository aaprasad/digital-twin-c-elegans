from analysis import get_results_numpy
import copy
import gym
import numpy as np
import os
from sim import position_func
import sys
from test import select_model
from test_vector_env import data_func, x_func, y_func
import torch
from virtual_nematode.envs.muscle_ellipsoid2d import (
    make_swimmer_weathervane_xml, make_swimmer_weathervane_fn, make_swimmer_weathervane_fixed_fn
)
from virtual_nematode.testers.tester import test_vector_env_func


def modify_all_mask(model, index):
    print('modify all mask, index', index)
    model.cell.w_c_mask[:, index] = False
    model.cell.w_c_mask[index, :] = False
    model.cell.w_g_mask[:, index] = False
    model.cell.w_g_mask[index, :] = False
    return model


def modify_chemical_mask(model, index):
    print('modify chemical mask, index', index)
    model.cell.w_c_mask[:, index] = False
    model.cell.w_c_mask[index, :] = False
    return model


def modify_electrical_mask(model, index):
    print('modify electrical mask, index', index)
    model.cell.w_g_mask[:, index] = False
    model.cell.w_g_mask[index, :] = False
    return model


def test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type, suffix='', device_id=None, id_list=None):
    print(ckpt_name)
    print('ablation', ablation_type, device_id)
    model = select_model(model_folder, model_name, ckpt_name)
    device = torch.device('cuda:{}'.format(device_id) if device_id is not None and torch.cuda.is_available() else 'cpu')
    save_folder_temp = os.path.join(save_folder, ablation_type + suffix)
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
        model_temp = model_temp.to(device)
        x, y = test_vector_env_func(env, model_temp, data_func, x_func, y_func, device, seed)
        print(x.shape, y.shape)
        np.savez(file_temp, x=x, y=y)
        get_results_numpy(x, y, max_episode_steps=max_episode_steps, sigma=5)


def main_random():
    """ env """
    env = gym.vector.AsyncVectorEnv(
        env_fns=[
            make_swimmer_weathervane_fn(
                xml_str, reset_noise_scale=0.6, distance=15, max_episode_steps=max_episode_steps,
                source=(0, 0), position_func=position_func
            ) for _ in range(num_envs)
        ]
    )
    print(env.action_space, env.observation_space)
    """ testing """
    test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=0, id_list=list(range(0, 160)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=1, id_list=list(range(160, 320)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=2, id_list=list(range(320, 469)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='chemical', device_id=3, id_list=list(range(0, 160)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='chemical', device_id=4, id_list=list(range(160, 320)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='chemical', device_id=5, id_list=list(range(320, 469)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='electrical', device_id=6, id_list=list(range(0, 235)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='electrical', device_id=7, id_list=list(range(235, 469)))


def main_random_all():
    """ env """
    env = gym.vector.AsyncVectorEnv(
        env_fns=[
            make_swimmer_weathervane_fn(
                xml_str, reset_noise_scale=0.6, distance=15, max_episode_steps=max_episode_steps,
                source=(0, 0), position_func=position_func
            ) for _ in range(num_envs)
        ]
    )
    print(env.action_space, env.observation_space)
    """ testing """
    test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=0, id_list=list(range(0, 59)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=1, id_list=list(range(59, 118)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=2, id_list=list(range(118, 177)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=3, id_list=list(range(177, 236)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=4, id_list=list(range(236, 295)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=5, id_list=list(range(295, 354)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=6, id_list=list(range(354, 413)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', device_id=7, id_list=list(range(413, 469)))


def main_fixed(pos=(10, 0), device_id=0):
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
    suffix = '_d{}'.format(distance)
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', suffix=suffix, device_id=0, id_list=list(range(0, 160)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', suffix=suffix, device_id=1, id_list=list(range(160, 320)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', suffix=suffix, device_id=2, id_list=list(range(320, 469)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='chemical', suffix=suffix, device_id=3, id_list=list(range(0, 160)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='chemical', suffix=suffix, device_id=4, id_list=list(range(160, 320)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='chemical', suffix=suffix, device_id=5, id_list=list(range(320, 469)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='electrical', suffix=suffix, device_id=6, id_list=list(range(0, 235)))
    # test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='electrical', suffix=suffix, device_id=7, id_list=list(range(235, 469)))

    if device_id == 0:
        id_list = list(range(0, 59))
    elif device_id == 1:
        id_list = list(range(59, 118))
    elif device_id == 2:
        id_list = list(range(118, 177))
    elif device_id == 3:
        id_list = list(range(177, 236))
    elif device_id == 4:
        id_list = list(range(236, 295))
    elif device_id == 5:
        id_list = list(range(295, 354))
    elif device_id == 6:
        id_list = list(range(354, 413))
    elif device_id == 7:
        id_list = list(range(413, 469))
    else:
        raise AssertionError
    test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', suffix=suffix, device_id=device_id, id_list=id_list)


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
    suffix = '_d{}'.format(distance)

    if device_id == 0 or device_id == 4:
        id_list = list(range(0, 117))
    elif device_id == 1 or device_id == 5:
        id_list = list(range(117, 234))
    elif device_id == 2 or device_id == 6:
        id_list = list(range(234, 351))
    elif device_id == 3 or device_id == 7:
        id_list = list(range(351, 469))
    else:
        raise AssertionError
    test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', suffix=suffix, device_id=device_id, id_list=id_list)


def main_fixed_6(pos=(10, 0), device_id=0):
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
    suffix = '_d{}'.format(distance)

    if device_id == 2:
        id_list = list(range(0, 78))
    elif device_id == 3:
        id_list = list(range(78, 156))
    elif device_id == 4:
        id_list = list(range(156, 234))
    elif device_id == 5:
        id_list = list(range(234, 312))
    elif device_id == 6:
        id_list = list(range(312, 390))
    elif device_id == 7:
        id_list = list(range(390, 469))
    else:
        raise AssertionError
    test(env, model_folder, model_name, ckpt_name, save_folder, ablation_type='all', suffix=suffix, device_id=device_id, id_list=id_list)


if __name__ == '__main__':
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
    """ env """
    xml_str = make_swimmer_weathervane_xml(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        source=(0, 0), density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    """ test """
    # model_name = 'li_conductance_gradient2'
    # model_name = 'lic41'
    model_name = 'lic62'
    # main_random()
    # main_random_all()
    # main_fixed(pos=(10, 0), device_id=device_id)
    main_fixed_4(pos=(10, 0), device_id=device_id)
    # main_fixed_6(pos=(10, 0), device_id=device_id)
