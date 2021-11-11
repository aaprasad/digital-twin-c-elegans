""" swimmer forward locomotion with vectorized env
vectorized environment: https://github.com/openai/gym/blob/master/gym/vector/vector_env.py
run multiple environments in parallel: https://github.com/openai/gym/blob/master/gym/vector/async_vector_env.py
run multiple environments serially: https://github.com/openai/gym/blob/master/gym/vector/sync_vector_env.py
"""

import gym
import numpy as np
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.models.forward import Forward


def action_func(model, step, observation, **kwargs):
    q = observation[:, 1:25]
    q_vel = observation[:, 28:52]
    action = model.step(step, q, q_vel)
    return action


def step_func(observation, **kwargs):
    com = observation[:, 52:54]
    return com


def done_func(result, **kwargs):
    displacement = np.linalg.norm(result[-1] - result[0], ord=2, axis=1)
    return displacement


def make_env(seed, **kwargs):
    """ https://github.com/openai/gym/blob/master/tests/vector/utils.py """
    def _make():
        env = make_swimmer(**kwargs)
        env.seed(seed)
        return env
    return _make


def simulate(env, model):
    result, results = [], None
    observation = env.reset()
    for step in range(10 ** 6):
        action = action_func(model=model, step=step, observation=observation)
        observation, reward, done, info = env.step(action)
        result.append(step_func(observation=observation))
        if done.all():
            results = done_func(result)
            break
    return results


if __name__ == '__main__':
    max_episode_steps = 2500
    trials = 100
    env_kwargs = {'n_bodies': 25, 'joint_range': '-100 100', 'body_len': 0.1, 'max_episode_steps': max_episode_steps, 'reset_noise_scale': 1.745}
    env_fns = [make_env(seed=i, **env_kwargs) for i in range(trials)]
    env = gym.vector.AsyncVectorEnv(env_fns)
    # env = gym.vector.SyncVectorEnv(env_fns)
    print(env.action_space)
    print(env.observation_space)
    env_dummy = make_env(seed=0, **env_kwargs)()
    model = Forward(dt=env_dummy.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    results = simulate(env, model)
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
