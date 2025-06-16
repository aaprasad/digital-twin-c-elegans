import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.models.muscle import ForwardPIDMuscle, WeathervanePIDMuscle, ShallowTurn


def simulate(env, model, action_func, seed=None, render=False):
    data = {'observation': [], 'action': []}
    env.seed(seed)
    observation = env.reset()
    model.reset()
    for step in range(10 ** 6):
        if render is True:
            env.render()
        action = action_func(model, step, observation)
        observation, reward, done, info = env.step(action)
        data['observation'].append(observation)
        data['action'].append(action)
        if done:
            break
    data = {key: np.array(value) for key, value in data.items()}
    return data


def process_data(path, start, end):
    data = np.load(path)
    plt.figure()
    plt.title('Joint Angle')
    sns.heatmap(data['observation'][start:end, 4:28], cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Time (steps)')
    plt.ylabel('Joint ID')
    plt.figure()
    plt.title('Muscle Action')
    sns.heatmap(data['action'][start:end, :], cmap='coolwarm', vmin=0, vmax=1)
    plt.xlabel('Time (steps)')
    plt.ylabel('Muscle ID')


def forward_crawl(env, render):
    def action_func(model, step, observation):
        action = model.step(step, q=observation[4:28])
        return action
    folder = os.path.join('video', 'forward_crawl')
    env = gym.wrappers.Monitor(env, directory=folder, force=True)
    model = ForwardPIDMuscle(
        dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    data = simulate(env, model, action_func, seed=None, render=render)
    np.savez(os.path.join(folder, 'data.npz'), **data)


def shallow_turn(env, render):
    def action_func(model, step, observation):
        action = model.step(step, q=observation[4:28], start_step=1250)
        return action
    folder = os.path.join('video', 'shallow_turn')
    env = gym.wrappers.Monitor(env, directory=folder, force=True)
    model = ShallowTurn(
        k_w=0.2, sigma=1., dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    data = simulate(env, model, action_func, seed=None, render=render)
    np.savez(os.path.join(folder, 'data.npz'), **data)


def gradual_turn(env, render):
    def action_func(model, step, observation):
        g_w = 0.05 if step >= 1250 else 0.
        action = model.step(step, q=observation[4:28], g_w=g_w)
        return action
    folder = os.path.join('video', 'gradual_turn')
    env = gym.wrappers.Monitor(env, directory=folder, force=True)
    model = WeathervanePIDMuscle(
        k_w=1, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    data = simulate(env, model, action_func, seed=None, render=render)
    np.savez(os.path.join(folder, 'data.npz'), **data)


def omega_turn():
    pass


def backward():
    pass


if __name__ == '__main__':
    env = make_swimmer(
        n_bodies=25, joint_range='-90 90', max_episode_steps=2500, reset_noise_scale=0.,
        density=1.2, viscosity=0.1, condim=3, friction='1 1'
    )
    print(env.action_space)
    print(env.observation_space)
    # forward_crawl(env, render=False)
    shallow_turn(env, render=False)
    # gradual_turn(env, render=False)
