import numpy as np
from sim import position_func, make_swimmer, step_func, done_func
from virtual_nematode.models.computational_model import ComputationalModelChemotaxis
from virtual_nematode.simulation import simulate


def action_func(model, step, observation, **kwargs):
    """ the sphere will be tracking center of mass/position of robot """
    q = observation[3:27]
    q_vel = observation[32:56]
    g_p = observation[67]
    g_w = observation[68]
    action = model.step(step, q, q_vel, g_p, g_w)
    _, position = position_func(observation)
    action = np.concatenate((action, position), axis=0)
    return action


if __name__ == '__main__':
    max_episode_steps = 2500
    env = make_swimmer(x=0, y=0, rgba='1 0 0 1', max_episode_steps=max_episode_steps)
    # env = gym.wrappers.Monitor(env, directory='video/swimmer_avoiding', force=True)
    print(env.action_space)
    print(env.observation_space)
    kwargs = {'backward': False, 'omega': False, 'weathervane': True, 'random_walk': False, 'weathervane_reverse': True}
    model = ComputationalModelChemotaxis(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2., **kwargs)
    results = simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False)
    print('{} trials: chemotaxis index mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
