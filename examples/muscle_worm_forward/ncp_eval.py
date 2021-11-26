from data import x_func as data_func
import gym
import os
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.envs.muscle_worm import make_swimmer
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model


def fully_connected(ckpt_name):
    """ results
    units = 200, batch_size = 256
        100 trials: com displacement mean 3.25 / 2500 steps
        1 trial: com displacement 3.42 / 2500 steps
    units = 150, batch_size = 512
        100 trials: com displacement mean 2.78 / 2500 steps
        1 trial: com displacement 2.95 / 2500 steps
    units = 100, batch_size = 1024
        100 trials: com displacement mean 2.38 / 2500 steps
        1 trial: com displacement 2.60 / 2500 steps
    units = 128, batch_size = 512
        100 trials: com displacement mean 3.02 / 2500 steps
        1 trial: com displacement 3.23 / 2500 steps
    """
    model_name = 'fully_connected'
    model = prepare_model(
        model_name, model_path=os.path.join(model_folder, ckpt_name),
        **{'units': 150, 'output_dim': 96, 'in_features': 193, 'output_mapping': 'affine'}
    )
    return model, model_name


def evaluate_all(start, end):
    """ online test each checkpoint once for evaluation """
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        model, _ = fully_connected(ckpt_name)
        print(ckpt_name, end=' ')
        _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps)
        torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


def evaluate(env, ckpt_name):
    """ online test once for evaluation and record video """
    model, _ = fully_connected(ckpt_name)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder), force=True)
    _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps)
    torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


def test(ckpt_name):
    """ online test multiple trials for testing """
    model, model_name = fully_connected(ckpt_name)
    tester(env, model, data_func, x_func, seed, max_episode_steps, model_folder, model_name, data_size=100)


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
    evaluate_all(start=0, end=300)
    # evaluate(env, ckpt_name='model.pt')
    # test(ckpt_name='model.pt')
