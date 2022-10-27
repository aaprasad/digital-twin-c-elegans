import numpy as np
import os
from sim import action_func
import torch
from virtual_nematode.data.simulation import generate_dataset
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.models.muscle import ForwardPIDMuscle


def x_func(observation, **kwargs):
    return observation[4:28]  # joint angles


def y_func(action, **kwargs):
    return action.tolist()


if __name__ == '__main__':
    input_size = 24  # joint angles
    data_size = 7000  # 7000, 3500, 1750
    max_episode_steps = 640  # 640, 1280, 2560
    reset_noise_scale = 0.6  # reset_noise_scale: 0.7->0.6
    seed = 7
    env = make_swimmer(
        n_bodies=25, joint_range='-90 90', max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale,
        density=1.2, viscosity=0.1, condim=3, friction='1 1'
    )
    action_size = env.action_space.shape[0]
    print(env.action_space)
    print(env.observation_space)
    # model = ForwardPIDMuscle(
    #     dt=env.dt, n=25, a=40 * np.pi / 180, freq=0.8, psi=0.07,
    #     kp=1, kd=np.array([0.15 + i * 0.002 for i in range(24)])
    # )
    model = ForwardPIDMuscle(
        dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    dataset = generate_dataset(env, model, action_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed)
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data_7000_640.pt')  # 'data_7000_640.pt', 'data_3500_1280.pt', 'data_1750_2560.pt'
