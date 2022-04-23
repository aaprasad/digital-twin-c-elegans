from data import x_func as data_func
import os
import torch
from virtual_nematode.envs.swimmer import make_chemotaxis_swimmers
from virtual_nematode.testers.chemotaxis import tester
from virtual_nematode.trainers.ncp import prepare_model


def position_func(observation, **kwargs):
    """ 2D center of mass and position of the first body segment """
    com, position = observation[52:54], observation[55:57]
    return com, position


def x_func(observation, **kwargs):
    """ accumulate concentrations along the path """
    concentration = [observation[58]]  # (1, )
    return concentration


def select_model(model_name, ckpt_name):
    if model_name == 'ctrnn':
        torch.set_default_dtype(torch.float64)
        kwargs = {'input_size': 48, 'hidden_size': 50, 'output_size': 24, 'feedback': True, 'readout': 'identity'}
    else:
        raise AssertionError('{} not exist'.format(model_name))
    model = prepare_model(model_name, model_path=os.path.join(model_folder, ckpt_name), **kwargs)
    return model


def test(model_name, start, end):
    """ online test multiple trials for testing with torch multiprocessing """
    envs = make_chemotaxis_swimmers(
        seed=seed, trials=trials, distance=distance, position_func=position_func, n_bodies=n_bodies,
        joint_range=joint_range, body_len=body_len, max_episode_steps=max_episode_steps,
        reset_noise_scale=reset_noise_scale, camera_name=None, return_func=False
    )
    envs = envs * data_size_per_trial
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        print(ckpt_name, end=' ')
        model = select_model(model_name, ckpt_name)
        tester(envs, model, data_func, x_func, seed, max_episode_steps, model_folder, model_name, disable=True)


if __name__ == '__main__':
    runs_folder = 'Dec16_16-02-23_h-10-176-50-34'
    model_folder = os.path.join('runs', runs_folder)
    if os.path.exists(model_folder) is False:
        raise ValueError('Invalid runs folder {}'.format(runs_folder))
    seed = 42
    reset_noise_scale = 1.745
    max_episode_steps = 3500
    trials = 100  # amount of envs with different source positions
    data_size_per_trial = 1  # amount of simulations per env
    distance = 15
    n_bodies = 25
    joint_range = '-100 100'
    body_len = 0.1
    data_path = os.path.join('data_weathervane', runs_folder)  # data folder for storing model action sequence output
    os.makedirs(data_path, exist_ok=True)
    test('ctrnn', start=47, end=48)
