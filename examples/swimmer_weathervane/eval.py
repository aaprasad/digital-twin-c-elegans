from data import x_func as data_func
import gym
import os
from sim import position_func
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.envs.swimmer import make_chemotaxis_swimmer
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model


def select_model(model_name, ckpt_name):
    if model_name == 'fully_connected':
        kwargs = {'units': 50, 'output_dim': 24, 'in_features': 50}
    elif model_name == 'ctrnn':
        torch.set_default_dtype(torch.float64)
        kwargs = {'input_size': 50, 'hidden_size': 50, 'output_size': 24, 'feedback': True, 'readout': 'identity'}
    else:
        raise AssertionError('{} not exist'.format(model_name))
    model = prepare_model(model_name, model_path=os.path.join(model_folder, ckpt_name), **kwargs)
    return model


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    model_folder = os.path.join('runs', runs_folder)
    if os.path.exists(model_folder) is False:
        raise ValueError('Invalid runs folder {}'.format(runs_folder))
    seed = 42
    reset_noise_scale = 1.745
    max_episode_steps = 3500
    trials = 10  # amount of envs with different source positions
    data_size_per_trial = 10  # amount of simulations per env
    data_path = os.path.join('data', runs_folder)  # data folder for storing model action sequence output
    os.makedirs(data_path, exist_ok=True)
    envs = make_chemotaxis_swimmer(
        seed=seed, trials=trials, distance=15, position_func=position_func, n_bodies=25, joint_range='-100 100', body_len=0.1,
        max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale, camera_name=None
    )
    evaluate('fully_connected', start=0, end=100)
    # test('fully_connected', start=0, end=100)
    # record('fully_connected', env, ckpt_name='model.pt')
