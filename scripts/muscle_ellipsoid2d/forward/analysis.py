import numpy as np
import torch


def get_results_numpy(x, y, **kwargs):
    """ x (trials, max_episode_steps, observation_length); y (trials, max_episode_steps, action_length) """
    max_episode_steps = kwargs.get('max_episode_steps')
    displacement_mean = np.linalg.norm(x[:, -1, 56:58] - x[:, 0, 56:58], ord=2, axis=1).mean()
    print('{} trial(s), com displacement mean {:.2f} / {} steps'.format(x.shape[0], displacement_mean, max_episode_steps))
    distance_mean = np.linalg.norm(x[:, 1:, 56:58] - x[:, 0:-1, 56:58], ord=2, axis=2).sum(axis=1).mean()
    print('{} trial(s), com distance mean {:.2f} / {} steps'.format(x.shape[0], distance_mean, max_episode_steps))


def get_results_torch(x, y, **kwargs):
    """ x (trials, max_episode_steps, observation_length); y (trials, max_episode_steps, action_length) """
    max_episode_steps = kwargs.get('max_episode_steps')
    displacement_mean = torch.linalg.norm(x[:, -1, 56:58] - x[:, 0, 56:58], ord=2, dim=1).mean().item()
    print('{} trial(s), com displacement mean {:.2f} / {} steps'.format(x.shape[0], displacement_mean, max_episode_steps))
    distance_mean = torch.linalg.norm(x[:, 1:, 56:58] - x[:, 0:-1, 56:58], ord=2, dim=2).sum(dim=1).mean().item()
    print('{} trial(s), com distance mean {:.2f} / {} steps'.format(x.shape[0], distance_mean, max_episode_steps))


def get_result_torch(x, y, **kwargs):
    """ x (max_episode_steps, observation_length), y (max_episode_steps, action_length) """
    max_episode_steps = kwargs.get('max_episode_steps')
    displacement = torch.linalg.norm(x[-1, 56:58] - x[0, 56:58], ord=2).item()
    print('com displacement {:.2f} / {} steps'.format(displacement, max_episode_steps))
    distance = torch.linalg.norm(x[1:, 56:58] - x[0:-1, 56:58], ord=2, dim=1).sum(dim=1).mean().item()
    print('com distance {:.2f} / {} steps'.format(distance, max_episode_steps))
