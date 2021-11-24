""" swimmer: forward locomotion
swimmer configuration
    n_bodies = 25
    receive muscle ctrl signal
    perform joint ctrl
"""

from gym_worm.wrappers.muscle_action import MuscleAction, JointAction
import numpy as np
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation import simulate


def action_func(model, step, observation, **kwargs):
    q = observation[1:25]
    q_vel = observation[28:52]
    action = model.step(step, q, q_vel)
    return action


def step_func(observation, **kwargs):
    com = observation[52:54]
    return com


def done_func(index, result, **kwargs):
    displacement = np.linalg.norm(np.array(result[-1]) - np.array(result[0]), ord=2)
    print('Trial {}: com displacement {:.2f}'.format(index + 1, displacement))
    return displacement


if __name__ == '__main__':
    """ results
    100 trials: com displacement mean 16.89 / 2500 steps
    """
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=max_episode_steps, reset_noise_scale=1.745
    )
    env = MuscleAction(env)  # ctrl: muscle -> joint
    env = JointAction(env)  # ctrl: joint -> muscle
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    results = simulate(env, model, action_func, step_func, done_func, seed=None, trials=100, render=False)
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
