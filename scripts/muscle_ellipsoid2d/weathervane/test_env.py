from analysis import get_results_torch, get_result_torch
import gymnasium as gym
import numpy as np
import os
from sim import x_func, position_func
import sys
from test import data_func, y_func, select_model, y_func1
import torch
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer_weathervane, make_swimmer_weathervane_fixed
from virtual_nematode.envs.swimmer import fick_uniform
from virtual_nematode.testers.tester import tester, test_func


def test_random():
    env = make_swimmer_weathervane(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        distance=15, source=(0, 0), position_func=position_func,  # distance = 3 * sigma
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    x_func_size = env.observation_space.shape[0]
    y_func_size = 95
    x, y = tester(env, model, data_func, x_func, y_func, x_func_size, y_func_size, seed, max_episode_steps, data_size=100)
    torch.save((x, y), os.path.join(save_folder, 'test100.pt'))
    get_results_torch(x, y, max_episode_steps=max_episode_steps, sigma=5)


def test_random_uniform():
    env = make_swimmer_weathervane(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        distance=15, source=(0, 0), position_func=position_func,  # distance = 3 * sigma
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic',
        distribution_func=fick_uniform  # uniform distribution of concentration=0
    )
    x_func_size = env.observation_space.shape[0]
    y_func_size = 95
    x, y = tester(env, model, data_func, x_func, y_func, x_func_size, y_func_size, seed, max_episode_steps, data_size=100)
    torch.save((x, y), os.path.join(save_folder, 'test100.uniform.pt'))
    get_results_torch(x, y, max_episode_steps=max_episode_steps, sigma=5)


def test_fixed(pos=(15, 0), y_function=y_func, y_func_size=95, suffix=''):
    env = make_swimmer_weathervane_fixed(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        pos=pos, source=(0, 0), position_func=position_func,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    x_func_size = env.observation_space.shape[0]
    x, y = tester(env, model, data_func, x_func, y_function, x_func_size, y_func_size, seed, max_episode_steps, data_size=100)
    distance = int(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
    torch.save((x, y), os.path.join(save_folder, 'test100_d{}{}.pt'.format(distance, suffix)))
    get_results_torch(x, y, max_episode_steps=max_episode_steps, sigma=5)


def test_once_fixed(pos=(15, 0), y_function=y_func, suffix='', angle=None, camera_name=None, camera_z=50):
    env = make_swimmer_weathervane_fixed(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        pos=pos, source=(0, 0), position_func=position_func,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic',
        angle=angle, camera_pos='-1.25 0 2.5', camera_name=camera_name, camera_z=camera_z
    )
    env = gym.wrappers.Monitor(env, directory=save_folder, force=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y = test_func(env, model, data_func, x_func, y_function)
    distance = int(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
    torch.save((x, y), os.path.join(save_folder, 'test1_d{}_a{:.2f}_{}_{}.pt'.format(distance, angle, suffix, camera_name)))
    get_result_torch(x, y, max_episode_steps=max_episode_steps, sigma=5)


def test_fixed_uniform(pos=(15, 0)):
    env = make_swimmer_weathervane_fixed(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        pos=pos, source=(0, 0), position_func=position_func,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic',
        distribution_func=fick_uniform  # uniform distribution of concentration=0
    )
    x_func_size = env.observation_space.shape[0]
    y_func_size = 95
    x, y = tester(env, model, data_func, x_func, y_func, x_func_size, y_func_size, seed, max_episode_steps, data_size=100)
    distance = int(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
    torch.save((x, y), os.path.join(save_folder, 'test100_d{}.uniform.pt'.format(distance)))
    get_results_torch(x, y, max_episode_steps=max_episode_steps, sigma=5)


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
    seed = 7
    max_episode_steps = 2500
    data_path = os.path.join('data', runs_folder)
    os.makedirs(data_path, exist_ok=True)
    """ model """
    # model_name = 'li_conductance_gradient2'
    model_name = 'lic62'
    model = select_model(model_folder, model_name, ckpt_name)
    """ chemotaxis env """
    # test_random()
    # test_fixed(pos=(15, 0))
    # test_fixed(pos=(10, 0))
    # test_fixed(pos=(5, 0))
    """ uniform env """
    # test_random_uniform()
    # test_fixed_uniform(pos=(0, 0))
    """ chemotaxis env: full state """
    # test_fixed(pos=(10, 0), y_function=y_func1, y_func_size=1033, suffix='_full')
    # test_fixed(pos=(15, 0), y_function=y_func1, y_func_size=1033, suffix='_full')
    # test_once_fixed(pos=(-10, 0), y_function=y_func1, suffix='_full', angle=np.pi/2.)
    test_once_fixed(pos=(-10, 0), y_function=y_func1, suffix='_full', angle=np.pi/2., camera_name='fixedcam', camera_z=32)
