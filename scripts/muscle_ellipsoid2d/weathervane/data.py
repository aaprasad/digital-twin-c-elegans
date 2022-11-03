import numpy as np
import os
from sim import action_func, position_func
import torch
from virtual_nematode.data.simulation import generate_dataset
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer_weathervane
from virtual_nematode.models.muscle import WeathervanePIDMuscle


def x_func(observation, **kwargs):
    # q = observation[4:28]
    # c = observation[62:63]
    # g = observation[63:64]
    # g_p = observation[64:65]
    # g_w = observation[65:66]
    return observation


def y_func(action, **kwargs):
    return action.tolist()


if __name__ == '__main__':
    input_size = 66  # x_func() size
    data_size = 7000
    max_episode_steps = 640
    reset_noise_scale = 0.6
    seed = 7
    env = make_swimmer_weathervane(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale,
        distance=15, source=(0, 0), position_func=position_func,  # distance = 3 * sigma
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    action_size = env.action_space.shape[0]
    print(env.action_space, env.observation_space)
    print(env.source)
    model = WeathervanePIDMuscle(
        k_w=1, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    dataset = generate_dataset(env, model, action_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed)
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data_7000_640.pt')
    x, y = dataset.tensors
    displacement_mean = torch.linalg.norm(x[:, -1, 56:58] - x[:, 0, 56:58], ord=2, dim=1).mean().item()
    print('{} trial(s), com displacement mean {:.2f} / {} steps'.format(x.shape[0], displacement_mean, max_episode_steps))
    distance_mean = torch.linalg.norm(x[:, 1:, 56:58] - x[:, 0:-1, 56:58], ord=2, dim=2).sum(dim=1).mean().item()
    print('{} trial(s), com distance mean {:.2f} / {} steps'.format(x.shape[0], distance_mean, max_episode_steps))
    chemotaxis_index_mean = x[:, :, 62].mean().item()
    print('{} trials: chemotaxis index mean {:.2f}'.format(x.shape[0], chemotaxis_index_mean))
    initial_concentration_mean = x[:, 0, 62].mean().item()
    print('{} trials: initial concentration mean {:.2f}'.format(x.shape[0], initial_concentration_mean))
