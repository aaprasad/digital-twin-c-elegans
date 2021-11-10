""" muscle worm: forward locomotion
joint range [-40, 40] degrees -> [-0.7, 0.7] rad
"""

from gym_worm.wrappers.muscle_action import JointAction
import numpy as np
from virtual_nematode.envs.muscle_worm import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation import simulate


def action_func(model, step, observation, **kwargs):
    q = observation[1:25]
    q_vel = observation[28:52]
    action = model.step(step, q, q_vel)
    return action


def step_func(observation, **kwargs):
    com = observation[244:246]
    return com


def done_func(index, result, **kwargs):
    displacement = np.linalg.norm(np.array(result[-1]) - np.array(result[0]), ord=2)
    print('Trial {}: com displacement {:.2f}'.format(index + 1, displacement))
    return displacement


if __name__ == '__main__':
    """ results
    100 trials: com displacement mean 10.07 / 2500 steps
    """
    max_episode_steps = 2500
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=0.7)
    env = JointAction(env)  # ctrl: joint -> muscle
    print(env.action_space)
    print(env.observation_space)
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=1.5)
    results = simulate(env, model, action_func, step_func, done_func, seed=None, trials=100, render=False)
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
