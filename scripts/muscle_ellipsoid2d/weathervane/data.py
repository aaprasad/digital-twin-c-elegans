import numpy as np
import os
from sim import action_func, position_func
import torch
from virtual_nematode.data.simulation import generate_dataset
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer_weathervane
from virtual_nematode.models.muscle import WeathervanePIDMuscle


def x_func(observation, **kwargs):
    q = observation[4:28]
    # c = [observation[62]]
    g = [observation[63]]
    # g_p = [observation[64]]
    # g_w = [observation[65]]
    return np.concatenate((q, g))


def y_func(action, **kwargs):
    return action.tolist()


if __name__ == '__main__':
    input_size = 24 + 1
    data_size = 7000
    max_episode_steps = 640
    reset_noise_scale = 0.6
    seed = 7
    env = make_swimmer_weathervane(
        n_bodies=25, joint_range='-90 90', max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale,
        distance=15, position_func=position_func, density=1.2, viscosity=0.1, condim=3, friction='1 1', source=(0, 0)
    )
    action_size = env.action_space.shape[0]
    print(env.action_space)
    print(env.observation_space)
    print(env.source)
    model = WeathervanePIDMuscle(
        k_w=1, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    dataset = generate_dataset(env, model, action_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed)
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data_7000_640.pt')
