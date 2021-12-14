from data import x_func as data_func
import gym
import os
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.envs.muscle_worm import make_swimmer
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model


def select_model(model_name, ckpt_name):
    if model_name == 'fully_connected':
        kwargs = {'units': 100, 'output_dim': 96, 'in_features': 192, 'output_mapping': 'affine'}
    else:
        raise AssertionError('{} not exist'.format(model_name))
    model = prepare_model(model_name, model_path=os.path.join(model_folder, ckpt_name), **kwargs)
    return model


def evaluate(model_name, start, end):
    """ online test each checkpoint once for evaluation """
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        model = select_model(model_name, ckpt_name)
        print(ckpt_name, end=' ')
        _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps)
        torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


def test(model_name, start, end):
    """ online test multiple trials for testing """
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        model = select_model(model_name, ckpt_name)
        tester(env, model, data_func, x_func, seed, max_episode_steps, model_folder, model_name, data_size=100)


def record(model_name, env, ckpt_name):
    """ online test once for evaluation and record video """
    model = select_model(model_name, ckpt_name)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder, ckpt_name), force=True)
    _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps)
    torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    model_folder = os.path.join('runs', runs_folder)
    if os.path.exists(model_folder) is False:
        raise ValueError('Invalid runs folder {}'.format(runs_folder))
    seed = 42
    reset_noise_scale = 0.7
    max_episode_steps = 2500
    data_path = os.path.join('data', runs_folder)  # data folder for storing model action sequence output
    os.makedirs(data_path, exist_ok=True)
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale)
    evaluate('fully_connected', start=0, end=100)
    # test('fully_connected', start=0, end=100)
    # record('fully_connected', env, ckpt_name='model.pt')
