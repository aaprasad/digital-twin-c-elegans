from analysis import get_results_numpy
import gymnasium as gym
import numpy as np
import os
from sim import action_func, x_func
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.models.muscle import ForwardPIDMuscle
from virtual_nematode.simulation import simulate


class TurnPIDMuscle(ForwardPIDMuscle):
    def step(self, step, q, **kwargs):
        g_w = 0.1
        if 32 <= step < 128:
            k_w = 1
        else:
            k_w = 0
        q_target = self.a * np.sin(self.omega * step * self.dt + self.phi) - k_w * g_w
        action = self._action(q, q_target)
        return action


if __name__ == '__main__':
    max_episode_steps = 160
    env = make_swimmer(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=max_episode_steps, reset_noise_scale=0.,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    directory = 'video/swimmer_turn_{}'.format(max_episode_steps)
    env = gym.wrappers.Monitor(env, directory=directory, force=True)
    print(env.action_space)
    print(env.observation_space)
    model = TurnPIDMuscle(
        dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    x, y = simulate(env, model, action_func, x_func, seed=None, trials=1, render=False)
    os.makedirs('data', exist_ok=True)
    np.savez(os.path.join(directory, 'simulate_turn.npz'), x=x, y=y)
    get_results_numpy(x, y, max_episode_steps=max_episode_steps)
