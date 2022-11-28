from analysis import get_result_torch
import copy
from matplotlib import pyplot as plt
import os
import seaborn as sns
from sim import x_func
import sys
from test import data_func, y_func1, select_model
import torch
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.testers.tester import single_tester


def modify_all_mask(model, index):
    print('modify all mask, index', index)
    model.cell.w_c_mask[:, index] = False
    model.cell.w_c_mask[index, :] = False
    model.cell.w_g_mask[:, index] = False
    model.cell.w_g_mask[index, :] = False
    model.cell.w_p_mask[:, index] = False
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


def single_test_single_ablation(env, model_folder, model_name, ckpt_name, save_folder):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    save_folder_temp = os.path.join(save_folder, 'all')
    # save_folder_temp = os.path.join(save_folder, 'chemical')
    # save_folder_temp = os.path.join(save_folder, 'electrical')
    os.makedirs(save_folder_temp, exist_ok=True)
    for i in range(469):
        path_temp = os.path.join(save_folder_temp, 'single_test.{}.pt'.format(i))
        if os.path.exists(path_temp):
            continue
        model_temp = copy.deepcopy(model)
        model_temp = modify_all_mask(model_temp, index=i)
        # model_temp = modify_chemical_mask(model_temp, index=i)
        # model_temp = modify_electrical_mask(model_temp, index=i)
        x, y = single_tester(env, model_temp, data_func, x_func, y_func1, seed)
        torch.save((x, y), path_temp)
        get_result_torch(x, y, max_episode_steps=max_episode_steps)


def single_test_by_degrees(env, model_folder, model_name, ckpt_name, save_folder, amount):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    # cell index sorted by chemical degrees
    w_c_mask = model.cell.w_c_mask.clone().detach()
    chemical_degrees = w_c_mask.sum(dim=0) + w_c_mask.sum(dim=1)
    chemical_indexes = torch.argsort(chemical_degrees, descending=True)
    # cell index sorted by electrical degrees
    w_g_mask = model.cell.w_g_mask.clone().detach()
    assert torch.all(w_g_mask == w_g_mask.T)
    electrical_degrees = w_g_mask.sum(dim=0)
    electrical_indexes = torch.argsort(electrical_degrees, descending=True)
    # abaltion
    save_folder_temp = os.path.join(save_folder, 'chemical_by_degrees')
    # save_folder_temp = os.path.join(save_folder, 'electrical_by_degrees')
    os.makedirs(save_folder_temp, exist_ok=True)
    for i in range(amount):
        model_temp = copy.deepcopy(model)
        print('ablation by degrees first {}'.format(i + 1))
        for j in range(i + 1):
            model_temp = modify_chemical_mask(model_temp, index=chemical_indexes[j])
            print(j, chemical_indexes[j], chemical_degrees[chemical_indexes[j]])
            # model_temp = modify_electrical_mask(model_temp, index=electrical_indexes[j])
            # print(j, electrical_indexes[j], electrical_degrees[electrical_indexes[j]])
        x, y = single_tester(env, model_temp, data_func, x_func, y_func1, seed)
        torch.save((x, y), os.path.join(save_folder_temp, 'single_test.pt'))
        get_result_torch(x, y, max_episode_steps=max_episode_steps)


def single_test_by_degrees_double_abalation(env, model_folder, model_name, ckpt_name, save_folder, amount):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    # cell index sorted by chemical degrees
    w_c_mask = model.cell.w_c_mask.clone().detach()
    chemical_degrees = w_c_mask.sum(dim=0) + w_c_mask.sum(dim=1)
    chemical_indexes = torch.argsort(chemical_degrees, descending=True)
    # cell index sorted by electrical degrees
    w_g_mask = model.cell.w_g_mask.clone().detach()
    assert torch.all(w_g_mask == w_g_mask.T)
    electrical_degrees = w_g_mask.sum(dim=0)
    electrical_indexes = torch.argsort(electrical_degrees, descending=True)
    # abaltion
    save_folder_temp = os.path.join(save_folder, 'chemical_by_degrees_double_ablation')
    # save_folder_temp = os.path.join(save_folder, 'electrical_by_degrees_double_ablation')
    os.makedirs(save_folder_temp, exist_ok=True)
    for i in range(amount):
        for j in range(i + 1, amount):
            print('ablation by degrees {}-{}'.format(i, j))
            model_temp = copy.deepcopy(model)
            model_temp = modify_chemical_mask(model_temp, index=chemical_indexes[i])
            model_temp = modify_chemical_mask(model_temp, index=chemical_indexes[j])
            print(i, chemical_indexes[i], chemical_degrees[chemical_indexes[i]])
            print(j, chemical_indexes[j], chemical_degrees[chemical_indexes[j]])
            # model_temp = modify_electrical_mask(model_temp, index=electrical_indexes[i])
            # model_temp = modify_electrical_mask(model_temp, index=electrical_indexes[j])
            # print(i, electrical_indexes[i], electrical_degrees[electrical_indexes[i]])
            # print(j, electrical_indexes[j], electrical_degrees[electrical_indexes[j]])
            x, y = single_tester(env, model_temp, data_func, x_func, y_func1, seed)
            torch.save((x, y), os.path.join(save_folder_temp, 'single_test.{}-{}.pt'.format(i, j)))
            get_result_torch(x, y, max_episode_steps=max_episode_steps)


def plot_3_mask(index, path, w_c_mask, w_g_mask, w_p_mask):
    plt.figure(figsize=(20, 5))
    # chemical
    plt.subplot(1, 3, 1)
    plt.title('Ablation {} Chemical Synapse Mask {}'.format(index, w_c_mask.sum().item()))
    sns.heatmap(w_c_mask, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Cell ID')
    # gap junction
    plt.subplot(1, 3, 2)
    plt.title('Ablation {} Gap Junction Mask {}'.format(index, w_g_mask.sum().item()))
    sns.heatmap(w_g_mask, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Cell ID')
    # proprioception
    plt.subplot(1, 3, 3)
    plt.title('Ablation {} Proprioception Input Mask {}'.format(index, w_p_mask.sum().item()))
    sns.heatmap(w_p_mask, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Joint ID')
    plt.savefig(path)
    plt.close()


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
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    """ testing """
    model_name = 'li_conductance'
    # single_test_single_ablation(env, model_folder, model_name, ckpt_name, save_folder)
    # single_test_by_degrees(env, model_folder, model_name, ckpt_name, video_folder, amount=10)
    single_test_by_degrees_double_abalation(env, model_folder, model_name, ckpt_name, save_folder, amount=10)
