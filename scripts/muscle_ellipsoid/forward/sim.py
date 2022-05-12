""" ellipsoid swimmer with tendon muscles
observation space: Box(-inf, inf, (304,), float64)
    [0:298]: Muscle-Ellipsoid-v0 observation space
    [298:301]: x-, y- and z-coordinates of the robot's center of mass (length, m)
    [301:304]: x-, y- and z-coordinates of the front tip (length, m)
"""

import numpy as np
from virtual_nematode.envs.muscle_ellipsoid import make_swimmer
from virtual_nematode.models.sinusoidal import SinusoidalMuscle
from virtual_nematode.simulation import simulate


def action_func(model, step, **kwargs):
    action = model.step(step)
    return action


def step_func(observation, **kwargs):
    com = observation[298:300]  # 2D center of mass
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
    max_episode_steps = 2500  # 0.04s/step, 100s in total
    env = make_swimmer(
        n_bodies=25, joint_range='-40 40', theta=np.pi / 12, max_episode_steps=max_episode_steps, reset_noise_scale=0.,
        density=4000, viscosity=0.1, condim=3, friction='0.1 1'
    )
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space)
    print(env.observation_space)
    model = SinusoidalMuscle(dt=env.dt, n=25, a=20 * np.pi / 180, freq=0.8, psi=0.05)
    results = simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False)
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
