from data import x_func as data_func_base
import gym
import os
from sim import position_func
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.envs.swimmer import make_chemotaxis_swimmers
from virtual_nematode.testers.chemotaxis import single_tester, tester
from virtual_nematode.trainers.ncp import prepare_model


def data_func(observation, **kwargs):
    data = data_func_base(observation, **kwargs)
    proprioception = data[0:48]
    gradient = data[49]
    gradient_positive = gradient if gradient > 0 else 0
    gradient_negative = abs(gradient) if gradient < 0 else 0
    return proprioception + [gradient_positive, gradient_negative]


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


def evaluate(model_name, start, end):
    """ online test once for evaluation """
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        model = select_model(model_name, ckpt_name)
        print(ckpt_name, end=' ')
        _, y = single_tester(envs[0], model, data_func, x_func, seed)
        torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


def test(model_name, start, end):
    """ online test multiple trials for testing """
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        print(ckpt_name, end=' ')
        model = select_model(model_name, ckpt_name)
        tester(envs, model, data_func, x_func, seed, max_episode_steps, model_folder, model_name, data_size_per_trial, disable=True)


def record(model_name, env, ckpt_name):
    """ online test once for evaluation and record video """
    model = select_model(model_name, ckpt_name)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder, ckpt_name), force=True)
    _, y = single_tester(env, model, data_func, x_func, seed)
    torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    model_folder = os.path.join('runs', runs_folder)
    if os.path.exists(model_folder) is False:
        raise ValueError('Invalid runs folder {}'.format(runs_folder))
    seed = 42
    reset_noise_scale = 1.745
    max_episode_steps = 3500
    trials = 100  # amount of envs with different source positions
    data_size_per_trial = 1  # amount of simulations per env
    data_path = os.path.join('data', runs_folder)  # data folder for storing model action sequence output
    os.makedirs(data_path, exist_ok=True)
    envs = make_chemotaxis_swimmers(
        seed=seed, trials=trials, distance=15, position_func=position_func, n_bodies=25, joint_range='-100 100', body_len=0.1,
        max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale, camera_name=None, return_func=False
    )
    evaluate('fully_connected', start=0, end=100)
    # test('fully_connected', start=0, end=100)
    # record('fully_connected', envs[0], ckpt_name='model.pt')
