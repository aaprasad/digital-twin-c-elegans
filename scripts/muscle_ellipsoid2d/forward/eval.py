from data import x_func as data_func
import gym
import os
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.connectomes.forward import get_kwargs
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.testers.forward import tester, single_tester, test_func2
from virtual_nematode.trainers.ncp import prepare_model
import worm_assets


def y_func(action, **kwargs):
    return action.tolist()


def y_func1(state, activation, action, **kwargs):
    return state.squeeze(dim=0).tolist() + activation.squeeze(dim=0).tolist() + action.squeeze(dim=0).tolist()


def select_model(model_folder, model_name, ckpt_name):
    if model_name.startswith('snn_forward'):
        kwargs = {
            'dt': 0.04, 'steps': 5,
            **get_kwargs(
                path=worm_assets.connectome_path(),
                polarity_path=worm_assets.polarity_path('Cook et al connectome.xls')
            )
        }
    elif model_name == 'ctrnn':
        kwargs = {
            'input_size': 24, 'hidden_size': 171, 'output_size': 95, 'feedback': True, 'readout': 'identity',
            'unfolds': 6, 'delta_t': 0.1, 'tau': 1
        }
    else:
        raise AssertionError('{} not exist'.format(model_name))
    model = prepare_model(model_name, model_path=os.path.join(model_folder, ckpt_name), **kwargs)
    return model


def evaluate(model_folder, model_name, ckpt_name):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    x, y = single_tester(env, model, data_func, x_func, y_func1, seed, max_episode_steps, test_func=test_func2)
    torch.save((x, y), os.path.join(data_path, ckpt_name))  # action sequence


def test(model_folder, model_name, ckpt_name):
    """ online test multiple trials for testing """
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    tester(env, model, data_func, x_func, y_func, seed, max_episode_steps, model_folder, model_name, data_size=100, test_func=test_func2)


def record(model_folder, model_name, env, ckpt_name):
    """ online test once for evaluation and record video """
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    video_folder = os.path.join('video', runs_folder, ckpt_name)
    env = gym.wrappers.Monitor(env, directory=video_folder, force=True)
    x, y = single_tester(env, model, data_func, x_func, y_func1, seed, max_episode_steps, test_func=test_func2)
    torch.save((x, y), os.path.join(video_folder, ckpt_name))  # action sequence


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    model_folder = os.path.join('runs', runs_folder)
    if os.path.exists(model_folder) is False:
        raise ValueError('Invalid runs folder {}'.format(runs_folder))
    seed = 42
    max_episode_steps = 2500
    data_path = os.path.join('data', runs_folder)
    os.makedirs(data_path, exist_ok=True)
    env = make_swimmer(
        n_bodies=25, joint_range='-90 90', max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        density=1.2, viscosity=0.1, condim=3, friction='1 1'
    )
    # evaluate(model_folder, 'snn_forward3', ckpt_name='model.pt')
    test(model_folder, 'snn_forward3', ckpt_name='model.pt')
    # record(model_folder, 'snn_forward3', env, ckpt_name='model.pt')
