import gym
from matplotlib import pyplot as plt
import os
import seaborn as sns
from sim import x_func
import sys
import torch
from virtual_nematode.connectomes.forward import get_kwargs
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer, make_swimmer_trapped
from virtual_nematode.testers.tester import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model
import worm_assets


def data_func(observation, **kwargs):
    return observation[4:28]  # joint angles


def y_func(**kwargs):
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
    x_func_size = env.observation_space.shape[0]
    y_func_size = 469 + 469 + 95
    x, y = tester(env, model, data_func, x_func, y_func, x_func_size, y_func_size, seed, max_episode_steps, data_size=100)
    # torch.save((x, y), os.path.join(save_folder, 'test.pt'))
    displacement_mean = torch.linalg.norm(x[:, -1, 56:58] - x[:, 0, 56:58], ord=2, dim=1).mean().item()
    print('com displacement mean {:.2f} / {} steps'.format(displacement_mean, max_episode_steps))
    distance_mean = torch.linalg.norm(x[:, 1:, 56:58] - x[:, 0:-1, 56:58], ord=2, dim=2).sum(dim=1).mean().item()
    print('com distance mean {:.2f} / {} steps'.format(distance_mean, max_episode_steps))


def single_test(env, model_folder, model_name, ckpt_name, save_folder):
    print(ckpt_name)
    model = select_model(model_folder, model_name, ckpt_name)
    x, y = single_tester(env, model, data_func, x_func, y_func, seed)
    torch.save((x, y), os.path.join(save_folder, 'single_test.pt'))
    displacement = torch.linalg.norm(x[-1, 56:58] - x[0, 56:58], ord=2).item()
    print('com displacement {:.2f} / {} steps'.format(displacement, max_episode_steps))
    distance = torch.linalg.norm(x[1:, 56:58] - x[0:-1, 56:58], ord=2, dim=2).sum(dim=1).mean().item()
    print('com distance {:.2f} / {} steps'.format(distance, max_episode_steps))


def record(env, model_folder, model_name, ckpt_name, video_folder):
    env = gym.wrappers.Monitor(env, directory=video_folder, force=True)
    single_test(env, model_folder, model_name, ckpt_name, save_folder=video_folder)


def plot_and_save_proprioception_mask(mask, path):
    plt.title('Exc/Inh Proprioception Input Mask')
    sns.heatmap(mask[0], cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Cell ID')
    plt.ylabel('Joint ID')
    plt.savefig(path)


def single_test_with_masked_input(env, model_folder, model_name, ckpt_name, save_folder, trapped_dims):
    print(ckpt_name)
    # load model
    model = select_model(model_folder, model_name, ckpt_name)
    # mask out the trapped input dimensions
    kwargs = get_kwargs(
        path=worm_assets.connectome_path(),
        polarity_path=worm_assets.polarity_path('Cook et al connectome.xls'),
        trapped_dims=trapped_dims
    )
    w_p_mask = kwargs['w_p_mask']
    w_p_n = w_p_mask.sum(dim=[0, 1])
    w_p_n[w_p_n == 0] = 1
    model.cell.w_p_mask.data = w_p_mask
    model.cell.w_p_n.data = w_p_n
    plot_and_save_proprioception_mask(model.cell.w_p_mask.data, path=os.path.join(save_folder, 'proprioception_mask.png'))
    # testing
    x, y = single_tester(env, model, data_func, x_func, y_func1, seed, max_episode_steps, test_func=test_func2)
    os.makedirs(save_folder, exist_ok=True)
    torch.save((x, y), os.path.join(save_folder, ckpt_name))


def record_trapped_experiment(model_folder, model_name, ckpt_name, body_index):
    """ trap experiment
    reset_noise_scale: set to 0.
    body_index: list of body index within [1, 2, ..., 25]
    muscle_index: list of muscle index within [1, 2, ..., 24]
    """
    muscle_index = body_index[0:-1]
    print(body_index, muscle_index)
    exp_name = '-'.join([str(i) for i in body_index]) if len(body_index) > 0 else 'unrestrained'
    save_folder = os.path.join('video', runs_folder, ckpt_name, exp_name)
    env = make_swimmer_trapped(
        n_bodies=25, joint_range='-90 90', max_episode_steps=max_episode_steps, reset_noise_scale=0.,
        density=1.2, viscosity=0.1, condim=3, friction='1 1', body_index=body_index, muscle_index=muscle_index
    )
    env = gym.wrappers.Monitor(env, directory=save_folder, force=True)
    single_test(env, model_folder, model_name, ckpt_name, save_folder)


def record_trapped_experiment_with_masked_input(model_folder, model_name, ckpt_name, body_index):
    """ trap experiment and mask out trapped input dimensions
    reset_noise_scale: set to 0.
    body_index: list of body index within [1, 2, ..., 25]
    muscle_index: list of muscle index within [1, 2, ..., 24]
    trapped_dims: list of trapped dims within [0, 1, ..., 23]
    """
    muscle_index = body_index[0:-1]
    trapped_dims = [i - 1 for i in muscle_index]
    print(body_index, muscle_index, trapped_dims)
    exp_name = '-'.join([str(i) for i in body_index]) + '-masked' if len(body_index) > 0 else 'unrestrained'
    save_folder = os.path.join('video', runs_folder, ckpt_name, exp_name)
    env = make_swimmer_trapped(
        n_bodies=25, joint_range='-90 90', max_episode_steps=max_episode_steps, reset_noise_scale=0.,
        density=1.2, viscosity=0.1, condim=3, friction='1 1', body_index=body_index, muscle_index=muscle_index
    )
    env = gym.wrappers.Monitor(env, directory=save_folder, force=True)
    single_test_with_masked_input(env, model_folder, model_name, ckpt_name, save_folder, trapped_dims)


if __name__ == '__main__':
    runs_folder = sys.argv[1]
    ckpt_name = sys.argv[2]  # 'model.pt'
    model_folder = os.path.join('runs', runs_folder)
    save_folder = os.path.join('data', runs_folder, ckpt_name)
    video_folder = os.path.join('video', runs_folder, ckpt_name)
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)
    assert os.path.exists(model_folder)
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
    # record_trapped_experiment(model_folder, model_name, ckpt_name, body_index=[10, 11, 12, 13, 14, 15, 16])
    # record_trapped_experiment_with_masked_input(model_folder, model_name, ckpt_name, body_index=[10, 11, 12, 13, 14, 15, 16])
