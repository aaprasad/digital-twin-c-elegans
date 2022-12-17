from analysis import get_results_torch
import numpy as np
import os
from sim import action_func, position_func
import torch
from virtual_nematode.data.simulation import generate_dataset
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer_weathervane, make_swimmer_weathervane_fixed
from virtual_nematode.models.muscle import WeathervanePIDMuscle, WeathervanePIDMuscleDelay


def x_func(observation, **kwargs):
    return observation


def y_func(action, **kwargs):
    return action.tolist()


def main():
    """ dataset """
    data_size, max_episode_steps = 7000, 640
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
    # model = WeathervanePIDMuscleDelay(
    #     k_w=1, delay_step=100, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
    #     kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
    #     kd=0.15
    # )
    dataset = generate_dataset(env, model, action_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed)
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data_7000_640.pt')
    # torch.save(dataset, 'data/data_7000_640_delay100.pt')
    x, y = dataset.tensors
    get_results_torch(x, y, max_episode_steps=max_episode_steps, sigma=5)


def main_random(k_w=1):
    """ simulation with random starting position """
    data_size, max_episode_steps = 1000, 2500
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
        k_w=k_w, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    dataset = generate_dataset(env, model, action_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed)
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data_1000_2500_kw{}.pt'.format(k_w))
    x, y = dataset.tensors
    get_results_torch(x, y, max_episode_steps=max_episode_steps, sigma=5)


def main_fixed(k_w=1, pos=(15, 0)):
    """ simulation with fixed starting position """
    data_size, max_episode_steps = 1000, 2500
    env = make_swimmer_weathervane_fixed(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.6,
        pos=pos, source=(0, 0), position_func=position_func,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    action_size = env.action_space.shape[0]
    print(env.action_space, env.observation_space)
    print(env.source)
    model = WeathervanePIDMuscle(
        k_w=k_w, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    dataset = generate_dataset(env, model, action_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed)
    os.makedirs('data', exist_ok=True)
    distance = int(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
    torch.save(dataset, 'data/data_1000_2500_kw{}_d{}.pt'.format(k_w, distance))
    x, y = dataset.tensors
    get_results_torch(x, y, max_episode_steps=max_episode_steps, sigma=5)


if __name__ == '__main__':
    input_size = 66  # x_func() size
    reset_noise_scale = 0.6
    seed = 7
    """ dataset """
    main()
    """ simulation """
    # main_random(k_w=1)
    # main_fixed(k_w=1, pos=(15, 0))
    # main_fixed(k_w=1, pos=(10, 0))
    # main_fixed(k_w=1, pos=(5, 0))
    # main_random(k_w=0)
    # main_fixed(k_w=0, pos=(15, 0))
    # main_fixed(k_w=0, pos=(10, 0))
    # main_fixed(k_w=0, pos=(5, 0))
    # main_fixed(k_w=0, pos=(0, 0))
