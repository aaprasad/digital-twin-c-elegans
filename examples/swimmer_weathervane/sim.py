""" swimmer: chemotaxis based on weathervane mechanism
observation[0:52]: qpos[2:27] and qvel[0:27]
    qpos[0:27]: x_pos, y_pos, ?, q[0:24]
    qvel[0:27]: x_vel, y_vel, ?, q_vel[0:24]
observation[52:55]: 3D center of mass
observation[55:58]: 3D position of the first body segment
observation[58:62]: concentration, g, g_p, g_w
"""

import numpy as np
from virtual_nematode.envs.swimmer import make_chemotaxis_swimmer
from virtual_nematode.models.computational_model import ComputationalModelChemotaxis
from virtual_nematode.simulation import simulate


def position_func(observation, **kwargs):
    """ 2D center of mass and position of the first body segment """
    com, position = observation[52:54], observation[55:57]
    return com, position


def action_func(model, step, observation, **kwargs):
    """
    model: mathematical model of controller
    return: action
    """
    q = observation[1:25]
    q_vel = observation[28:52]
    g_p, g_w = observation[60], observation[61]
    action = model.step(step, q, q_vel, g_p, g_w)
    return action


def step_func(observation, **kwargs):
    """ accumulate concentrations along the path """
    concentration = observation[58]
    return concentration


def done_func(index, result, **kwargs):
    """ calculate chemotaxis index with concentrations along the path """
    chemotaxis_index = np.mean(result)
    print('Trial {}: chemotaxis index {:.2f}'.format(index + 1, chemotaxis_index))
    return chemotaxis_index


if __name__ == '__main__':
    """ results
    100 trials: chemotaxis index mean ?
    """
    trails = 1
    envs = make_chemotaxis_swimmer(
        seed=101, trials=trails, distance=15, position_func=position_func, n_bodies=25, joint_range='-100 100', body_len=0.1,
        max_episode_steps=2500, reset_noise_scale=1.745
    )
    print(envs[0].action_space, envs[0].observation_space)
    kwargs = {'backward': False, 'omega': False, 'weathervane': True, 'random_walk': False, 'weathervane_reverse': False}
    results = []
    for i in range(trails):
        model = ComputationalModelChemotaxis(dt=envs[i].dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2., **kwargs)
        result = simulate(envs[i], model, action_func, step_func, done_func, seed=None, trials=1, render=False)
        results += result
    print('{} trials: chemotaxis index mean {:.2f}'.format(len(results), np.mean(results)))
