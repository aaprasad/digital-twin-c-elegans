from data import x_func as data_func
import gym
import os
from sim import step_func as x_func
import sys
import torch
from virtual_nematode.connectomes.forward import get_kwargs
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer, make_swimmer_trapped
from virtual_nematode.testers.forward import tester, single_tester, test_func2
from virtual_nematode.trainers.ncp import prepare_model
import worm_assets


def y_func(action, **kwargs):
    """ stats: action """
    return action.tolist()


def y_func1(state, activation, action, **kwargs):
    """ stats: state, activation, action """
    return state.squeeze(dim=0).tolist() + activation.squeeze(dim=0).tolist() + action.tolist()


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


def test(model_folder, model_name, ckpt_name):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    tester(env, model, data_func, x_func, y_func, seed, max_episode_steps, model_folder, model_name, data_size=100, test_func=test_func2)


def single_test(env, model_folder, model_name, ckpt_name, save_folder):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    x, y = single_tester(env, model, data_func, x_func, y_func1, seed, max_episode_steps, test_func=test_func2)
    os.makedirs(save_folder, exist_ok=True)
    torch.save((x, y), os.path.join(save_folder, ckpt_name))


def record(env, model_folder, model_name, ckpt_name):
    save_folder = os.path.join('video', runs_folder, ckpt_name)
    env = gym.wrappers.Monitor(env, directory=save_folder, force=True)
    single_test(env, model_folder, model_name, ckpt_name, save_folder)


def record_trapped_experiment(model_folder, model_name, ckpt_name, body_index):
    """ trap experiment
    reset_noise_scale = 0.
    """
    print(body_index)
    save_folder = os.path.join('video', runs_folder, ckpt_name, '-'.join([str(i) for i in body_index]))
    env = make_swimmer_trapped(
        n_bodies=25, joint_range='-90 90', max_episode_steps=max_episode_steps, reset_noise_scale=0.,
        density=1.2, viscosity=0.1, condim=3, friction='1 1', body_index=body_index
    )
    env = gym.wrappers.Monitor(env, directory=save_folder, force=True)
    single_test(env, model_folder, model_name, ckpt_name, save_folder)


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    model_folder = os.path.join('runs', runs_folder)
    if os.path.exists(model_folder) is False:
        raise ValueError('Invalid runs folder {}'.format(runs_folder))
    seed = 42
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range='-90 90', max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        density=1.2, viscosity=0.1, condim=3, friction='1 1'
    )
    test(model_folder, 'snn_forward3', ckpt_name='model.pt')
    # single_test(env, model_folder, 'snn_forward3', ckpt_name='model.pt', save_folder=os.path.join('data', runs_folder))
    # record(env, model_folder, 'snn_forward3', ckpt_name='model.pt')
    # record_trapped_experiment(model_folder, 'snn_forward3', ckpt_name='model.pt', body_index=[10, 11, 12, 13, 14, 15, 16])
