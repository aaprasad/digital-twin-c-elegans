import numpy as np
import torch


def get_results_numpy(x, y, **kwargs):
    max_episode_steps = kwargs.get('max_episode_steps')
    displacement_mean = np.linalg.norm(x[:, -1, 56:58] - x[:, 0, 56:58], ord=2, axis=1).mean()
    print('{} trial(s), com displacement mean {:.2f} / {} steps'.format(x.shape[0], displacement_mean, max_episode_steps))
    distance_mean = np.linalg.norm(x[:, 1:, 56:58] - x[:, 0:-1, 56:58], ord=2, axis=2).sum(axis=1).mean()
    print('{} trial(s), com distance mean {:.2f} / {} steps'.format(x.shape[0], distance_mean, max_episode_steps))
    chemotaxis_index_mean = x[:, :, 62].mean()
    print('{} trials: chemotaxis index mean {:.2f}'.format(x.shape[0], chemotaxis_index_mean))
    initial_concentration_mean = x[:, 0, 62].mean()
    print('{} trials: initial concentration mean {:.2f}'.format(x.shape[0], initial_concentration_mean))


def get_results_torch(x, y, **kwargs):
    max_episode_steps = kwargs.get('max_episode_steps')
    displacement_mean = torch.linalg.norm(x[:, -1, 56:58] - x[:, 0, 56:58], ord=2, dim=1).mean().item()
    print('{} trial(s), com displacement mean {:.2f} / {} steps'.format(x.shape[0], displacement_mean, max_episode_steps))
    distance_mean = torch.linalg.norm(x[:, 1:, 56:58] - x[:, 0:-1, 56:58], ord=2, dim=2).sum(dim=1).mean().item()
    print('{} trial(s), com distance mean {:.2f} / {} steps'.format(x.shape[0], distance_mean, max_episode_steps))
    chemotaxis_index_mean = x[:, :, 62].mean().item()
    print('{} trials: chemotaxis index mean {:.2f}'.format(x.shape[0], chemotaxis_index_mean))
    initial_concentration_mean = x[:, 0, 62].mean().item()
    print('{} trials: initial concentration mean {:.2f}'.format(x.shape[0], initial_concentration_mean))
