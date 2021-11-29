from data import x_func as data_func
import gym
import os
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model


def fully_connected(ckpt_name):
    """ results
    units = 100
        100 trials: com displacement mean 21.76 / 2500 steps
        1 trial: com displacement 21.60 / 2500 steps
    units = 72
        100 trials: com displacement mean 21.75 / 2500 steps
        1 trial: com displacement 21.40 / 2500 steps
    units = 150
        100 trials: com displacement mean 2.81 / 2500 steps
        1 trial: com displacement 2.60 / 2500 steps
    """
    model_name = 'fully_connected'
    model = prepare_model(
        model_name, model_path=os.path.join(model_folder, ckpt_name),
        **{'units': 50, 'output_dim': 24, 'in_features': 48}
    )
    return model, model_name


def ncp(ckpt_name):
    """ results
    100 trials: com displacement mean 6.90 / 2500 steps
    1 trial: com displacement 6.73 / 2500 steps
    """
    model_name = 'ncp'
    model = prepare_model(
        model_name, model_path=os.path.join(model_folder, ckpt_name),
        **{
            'in_features': 48, 'inter_neurons': 24, 'command_neurons': 48, 'motor_neurons': 24, 'sensory_fanout': 24,
            'inter_fanout': 48, 'recurrent_command_synapses': 48, 'motor_fanin': 48
        }
    )
    return model, model_name


def evaluate_all(start, end):
    """ online test each checkpoint once for evaluation """
    for i in range(start, end):
        ckpt_name = 'model{}.pt'.format(i)
        model, _ = fully_connected(ckpt_name)
        # model, _ = ncp(ckpt_name)
        print(ckpt_name, end=' ')
        _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps)
        torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


def test_once(env, ckpt_name):
    """ online test once for evaluation and record video """
    model, _ = fully_connected(ckpt_name)
    # model, _ = ncp(ckpt_name)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder, ckpt_name), force=True)
    _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps)
    torch.save(y, os.path.join(data_path, ckpt_name))  # action sequence


def test(ckpt_name):
    """ online test multiple trials for testing """
    model, model_name = fully_connected(ckpt_name)
    # model, model_name = ncp(ckpt_name)
    tester(env, model, data_func, x_func, seed, max_episode_steps, model_folder, model_name, data_size=100)


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    model_folder = os.path.join('runs', runs_folder)
    if os.path.exists(model_folder) is False:
        raise ValueError('Invalid runs folder {}'.format(runs_folder))
    seed = 42
    reset_noise_scale = 1.745
    max_episode_steps = 2500
    data_path = os.path.join('data', runs_folder)  # data folder for storing model action sequence output
    os.makedirs(data_path, exist_ok=True)
    env = make_swimmer(
        n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=max_episode_steps,
        reset_noise_scale=reset_noise_scale
    )
    evaluate_all(start=0, end=100)
    # test_once(env, ckpt_name='model.pt')
    # test(ckpt_name='model.pt')
