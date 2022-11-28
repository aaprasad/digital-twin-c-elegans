from analysis import get_results_torch, get_result_torch
import gym
import os
from sim import x_func
import sys
import torch
from virtual_nematode.connectomes.forward import get_kwargs
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.testers.tester import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model
import worm_assets


def data_func(observation, **kwargs):
    return observation[4:28]  # joint angles


def y_func(**kwargs):
    """ stats: action """
    # state = kwargs.get('state')
    # activation = kwargs.get('activation')
    action = kwargs.get('action')
    data = action.tolist()
    # data = state.squeeze(dim=0).tolist() + activation.squeeze(dim=0).tolist() + action.tolist()
    return data


def y_func1(**kwargs):
    """ stats: action """
    state = kwargs.get('state')
    activation = kwargs.get('activation')
    action = kwargs.get('action')
    # data = action.tolist()
    data = state.squeeze(dim=0).tolist() + activation.squeeze(dim=0).tolist() + action.tolist()
    return data


def select_model(model_folder, model_name, ckpt_name):
    if model_name.startswith('snn_forward') or model_name.startswith('li'):
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


def test(model_folder, model_name, ckpt_name, save_folder):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    # model.cell.modify()
    x_func_size = env.observation_space.shape[0]
    y_func_size = 95  # 469 + 469 + 95
    x, y = tester(env, model, data_func, x_func, y_func, x_func_size, y_func_size, seed, max_episode_steps, data_size=100)
    torch.save((x, y), os.path.join(save_folder, 'test100.pt'))
    get_results_torch(x, y, max_episode_steps=max_episode_steps)


def single_test(env, model_folder, model_name, ckpt_name, save_folder):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    # model = modify_chemical_mask(model, n)
    x, y = single_tester(env, model, data_func, x_func, y_func1, seed)
    torch.save((x, y), os.path.join(save_folder, 'test20221126', 'single_test.pt'))
    get_result_torch(x, y, max_episode_steps=max_episode_steps)


def record(env, model_folder, model_name, ckpt_name, video_folder):
    video_folder = os.path.join(video_folder, 'gap')
    env = gym.wrappers.Monitor(env, directory=video_folder, force=True)
    single_test(env, model_folder, model_name, ckpt_name, save_folder=video_folder)


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
    seed = 1024
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    """ testing """
    model_name = 'li_conductance'
    test(model_folder, model_name, ckpt_name, save_folder)
    # single_test(env, model_folder, model_name, ckpt_name, save_folder)
    # record(env, model_folder, model_name, ckpt_name, video_folder)
