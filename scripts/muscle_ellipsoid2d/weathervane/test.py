from data import x_func as data_func
from data import y_func
import gym
import os
from sim import position_func
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.connectomes.weathervane import get_kwargs
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer_weathervane
from virtual_nematode.testers.weathervane import single_tester, tester
from virtual_nematode.trainers.ncp import prepare_model
import worm_assets


def select_model(model_folder, model_name, ckpt_name):
    if model_name.startswith('snn_weathervane'):
        kwargs = {
            'dt': 0.04, 'steps': 5,
            **get_kwargs(
                path=worm_assets.connectome_path(),
                polarity_path=worm_assets.polarity_path('Cook et al connectome.xls')
            )
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    model = prepare_model(model_name, model_path=os.path.join(model_folder, ckpt_name), **kwargs)
    return model


def test(model_folder, model_name, ckpt_name):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    tester(env, model, data_func, x_func, y_func, seed, max_episode_steps, model_folder, model_name, data_size=100)


def record(model_folder, model_name, env, ckpt_name):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder, ckpt_name), force=True)
    x, y = single_tester(env, model, data_func, x_func, y_func, seed, max_episode_steps)
    torch.save((x, y), os.path.join(data_path, ckpt_name))  # concentration, action sequence


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    model_folder = os.path.join('runs', runs_folder)
    seed = 43
    max_episode_steps = 2500
    data_path = os.path.join('data', runs_folder)
    os.makedirs(data_path, exist_ok=True)
    env = make_swimmer_weathervane(
        n_bodies=25, joint_range='-90 90', max_episode_steps=max_episode_steps, reset_noise_scale=0.6, distance=15,
        position_func=position_func, density=1.2, viscosity=0.1, condim=3, friction='1 1', source=(0, 0)
    )
    test(model_folder, 'snn_weathervane3', ckpt_name='model.pt')
    # record(model_folder, 'snn_weathervane3', env, ckpt_name='model.pt')
