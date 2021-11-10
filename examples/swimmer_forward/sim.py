""" swimmer: forward locomotion
swimmer configuration
    joint size = 0.1 <- radius = 0.04mm
    whole body len = 2.5 <- length = 1mm
    joint amount = 24 <- amount of muscles in one quadrant = 24
    n_bodies = 25
    body len = 0.1
observation space: (52,) including qpos[2:27], qvel[0:25],
    qpos[0:27]: x_pos, y_pos, ?, q[0:24]
    qvel[0:27]: x_vel, y_vel, ?, q_vel[0:24]
"""

import numpy as np
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation.forward import simulate


def model_kwargs_func(observation, **kwargs):
    return {'q': observation[1:25], 'q_vel': observation[28:52]}


def step_func(observation, **kwargs):
    """
    return: 2D center of mass
    """
    com = observation[52:54]
    return com


def done_func(index, result, **kwargs):
    """
    result: list of com in 1 simulation
    return: displacement in this simulation
    """
    displacement = np.linalg.norm(np.array(result[-1]) - np.array(result[0]), ord=2)
    print('Trial {}: com displacement {:.2f}'.format(index + 1, displacement))
    return displacement


if __name__ == '__main__':
    """ results
    100 trials: com displacement mean 16.91 / 2500 steps
    """
    max_episode_steps = 2500
    env = make_swimmer(n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=max_episode_steps, reset_noise_scale=1.745)
    print(env.action_space)
    print(env.observation_space)
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    results = simulate(env, model, model_kwargs_func, step_func, done_func, seed=None, trials=100, render=False)
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
