import numpy as np
import torch


def get_results_numpy(x, y, **kwargs):
    max_episode_steps = kwargs.get('max_episode_steps')
    displacement_mean = np.linalg.norm(x[:, -1, 56:58] - x[:, 0, 56:58], ord=2, axis=1).mean()
    print('{} trial(s), com displacement mean {:.2f} / {} steps'.format(x.shape[0], displacement_mean, max_episode_steps))
    distance_mean = np.linalg.norm(x[:, 1:, 56:58] - x[:, 0:-1, 56:58], ord=2, axis=2).sum(axis=1).mean()
    print('{} trial(s), com distance mean {:.2f} / {} steps'.format(x.shape[0], distance_mean, max_episode_steps))
    chemotaxis_index_mean = x[:, :, 62].mean()
    print('{} trial(s), chemotaxis index mean {:.2f}'.format(x.shape[0], chemotaxis_index_mean))
    initial_concentration_mean = x[:, 0, 62].mean()
    print('{} trial(s), initial concentration mean {:.2f}'.format(x.shape[0], initial_concentration_mean))


def get_results_torch(x, y, **kwargs):
    max_episode_steps = kwargs.get('max_episode_steps')
    displacement_mean = torch.linalg.norm(x[:, -1, 56:58] - x[:, 0, 56:58], ord=2, dim=1).mean().item()
    print('{} trial(s), com displacement mean {:.2f} / {} steps'.format(x.shape[0], displacement_mean, max_episode_steps))
    distance_mean = torch.linalg.norm(x[:, 1:, 56:58] - x[:, 0:-1, 56:58], ord=2, dim=2).sum(dim=1).mean().item()
    print('{} trial(s), com distance mean {:.2f} / {} steps'.format(x.shape[0], distance_mean, max_episode_steps))
    chemotaxis_index_mean = x[:, :, 62].mean().item()
    print('{} trial(s), chemotaxis index mean {:.2f}'.format(x.shape[0], chemotaxis_index_mean))
    initial_concentration_mean = x[:, 0, 62].mean().item()
    print('{} trial(s), initial concentration mean {:.2f}'.format(x.shape[0], initial_concentration_mean))


def get_result_torch(x, y, **kwargs):
    max_episode_steps = kwargs.get('max_episode_steps')
    displacement = torch.linalg.norm(x[-1, 56:58] - x[0, 56:58], ord=2).item()
    print('com displacement {:.2f} / {} steps'.format(displacement, max_episode_steps))
    distance = torch.linalg.norm(x[1:, 56:58] - x[0:-1, 56:58], ord=2, dim=1).sum(dim=1).mean().item()
    print('com distance {:.2f} / {} steps'.format(distance, max_episode_steps))
    chemotaxis_index = x[:, 62].mean().item()
    print('chemotaxis index {:.2f}'.format(chemotaxis_index))
    initial_concentration = x[0, 62].item()
    print('initial concentration {:.2f}'.format(initial_concentration))
