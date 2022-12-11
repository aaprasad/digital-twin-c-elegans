import numpy as np
import torch


def get_results_numpy(x, y, **kwargs):
    """ x (trials, max_episode_steps, observation_length); y (trials, max_episode_steps, action_length) """
    max_episode_steps = kwargs.get('max_episode_steps')
    sigma = kwargs.get('sigma')
    displacement_mean = np.linalg.norm(x[:, -1, 56:58] - x[:, 0, 56:58], ord=2, axis=1).mean()
    print('{} trial(s), com displacement mean {:.2f} / {} steps'.format(x.shape[0], displacement_mean, max_episode_steps))
    distance_mean = np.linalg.norm(x[:, 1:, 56:58] - x[:, 0:-1, 56:58], ord=2, axis=2).sum(axis=1).mean()
    print('{} trial(s), com distance mean {:.2f} / {} steps'.format(x.shape[0], distance_mean, max_episode_steps))
    chemotaxis_index_mean = x[:, :, 62].mean()
    print('{} trial(s), chemotaxis index mean {:.2f}'.format(x.shape[0], chemotaxis_index_mean))
    initial_concentration_mean = x[:, 0, 62].mean()
    print('{} trial(s), initial concentration mean {:.2f}'.format(x.shape[0], initial_concentration_mean))
    com = np.linalg.norm(x[:, :, 56:58], ord=2, axis=2)
    com_1sigma = np.any(com < 1 * sigma, axis=1).sum() / x.shape[0]
    print('{} trial(s), inside 1 sigma {:.4f}'.format(x.shape[0], com_1sigma))
    com_2sigma = np.any(com < 2 * sigma, axis=1).sum() / x.shape[0]
    print('{} trial(s), inside 2 sigma {:.4f}'.format(x.shape[0], com_2sigma))
    com_3sigma = np.any(com < 3 * sigma, axis=1).sum() / x.shape[0]
    print('{} trial(s), inside 3 sigma {:.4f}'.format(x.shape[0], com_3sigma))


def get_results_torch(x, y, **kwargs):
    """ x (trials, max_episode_steps, observation_length); y (trials, max_episode_steps, action_length) """
    max_episode_steps = kwargs.get('max_episode_steps')
    sigma = kwargs.get('sigma')
    displacement_mean = torch.linalg.norm(x[:, -1, 56:58] - x[:, 0, 56:58], ord=2, dim=1).mean().item()
    print('{} trial(s), com displacement mean {:.2f} / {} steps'.format(x.shape[0], displacement_mean, max_episode_steps))
    distance_mean = torch.linalg.norm(x[:, 1:, 56:58] - x[:, 0:-1, 56:58], ord=2, dim=2).sum(dim=1).mean().item()
    print('{} trial(s), com distance mean {:.2f} / {} steps'.format(x.shape[0], distance_mean, max_episode_steps))
    chemotaxis_index_mean = x[:, :, 62].mean().item()
    print('{} trial(s), chemotaxis index mean {:.2f}'.format(x.shape[0], chemotaxis_index_mean))
    initial_concentration_mean = x[:, 0, 62].mean().item()
    print('{} trial(s), initial concentration mean {:.2f}'.format(x.shape[0], initial_concentration_mean))
    com = torch.linalg.norm(x[:, :, 56:58], ord=2, dim=2)
    com_1sigma = torch.any(com < 1 * sigma, dim=1).sum().item() / x.shape[0]
    print('{} trial(s), inside 1 sigma {:.4f}'.format(x.shape[0], com_1sigma))
    com_2sigma = torch.any(com < 2 * sigma, dim=1).sum().item() / x.shape[0]
    print('{} trial(s), inside 2 sigma {:.4f}'.format(x.shape[0], com_2sigma))
    com_3sigma = torch.any(com < 3 * sigma, dim=1).sum().item() / x.shape[0]
    print('{} trial(s), inside 3 sigma {:.4f}'.format(x.shape[0], com_3sigma))


def get_result_torch(x, y, **kwargs):
    """ x (max_episode_steps, observation_length), y (max_episode_steps, action_length) """
    max_episode_steps = kwargs.get('max_episode_steps')
    sigma = kwargs.get('sigma')
    displacement = torch.linalg.norm(x[-1, 56:58] - x[0, 56:58], ord=2).item()
    print('com displacement {:.2f} / {} steps'.format(displacement, max_episode_steps))
    distance = torch.linalg.norm(x[1:, 56:58] - x[0:-1, 56:58], ord=2, dim=1).sum().item()
    print('com distance {:.2f} / {} steps'.format(distance, max_episode_steps))
    chemotaxis_index = x[:, 62].mean().item()
    print('chemotaxis index {:.2f}'.format(chemotaxis_index))
    initial_concentration = x[0, 62].item()
    print('initial concentration {:.2f}'.format(initial_concentration))
    com = torch.linalg.norm(x[:, 56:58], ord=2, dim=1)
    com_1sigma = torch.any(com < 1 * sigma)
    print('inside 1 sigma {}'.format(com_1sigma))
    com_2sigma = torch.any(com < 2 * sigma)
    print('inside 2 sigma {}'.format(com_2sigma))
    com_3sigma = torch.any(com < 3 * sigma)
    print('inside 3 sigma {}'.format(com_3sigma))
