from analysis import get_results_numpy
import copy
import gym
import numpy as np
import os
from sim import position_func
import sys
from test import select_model
import torch
from tqdm import tqdm
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer_weathervane_xml, make_swimmer_weathervane_fn
from virtual_nematode.testers.tester import test_vector_env_func


def data_func(observation, **kwargs):
    q = observation[:, 4:28]
    g = observation[:, 63:64]
    return np.concatenate((q, g), axis=1)


def x_func(observation):
    return observation


def y_func(**kwargs):
    # state = kwargs.get('state')
    # activation = kwargs.get('activation')
    action = kwargs.get('action')
    data = action
    # data = np.concatenate((state, activation, action), axis=1)
    return data


def test(model_folder, model_name, ckpt_name, save_folder, device_id=None):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    device = torch.device('cuda:{}'.format(device_id) if device_id is not None and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for i in tqdm(range(1)):
        model_temp = copy.deepcopy(model)
        x, y = test_vector_env_func(env, model_temp, data_func, x_func, y_func, device, seed)
        print(x.shape, y.shape)
        np.savez(os.path.join(save_folder, 'test100.{}.npz'.format(i)), x=x, y=y)
        get_results_numpy(x, y, max_episode_steps=max_episode_steps, sigma=5)


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
    data_path = os.path.join('data', runs_folder)
    os.makedirs(data_path, exist_ok=True)
    """ env """
    xml_str = make_swimmer_weathervane_xml(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        source=(0, 0), density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
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
    model_name = 'li_conductance_gradient2'
    test(model_folder, model_name, ckpt_name, save_folder, device_id=0)
