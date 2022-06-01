"""
observation space: Box(-inf, inf, (350,), float64)
    [0:341]: Muscle-Ellipsoid2D-v0 observation space
    [341:344]: x-, y- and z-coordinates of the robot's center of mass (length, m)
    [344:347]: x-, y- and z-coordinates of the front tip (length, m)
    [347:350]: x-, y- and z-coordinates of the 17th body (length, m)
"""

import numpy as np
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer
from virtual_nematode.models.muscle import ForwardMuscle, ForwardPIDMuscle
from virtual_nematode.models.sinusoidal import SinusoidalMuscle
from virtual_nematode.simulation import simulate


def action_func(model, step, observation, **kwargs):
    q = observation[4:28]
    q_vel = observation[32:56]
    pos1 = observation[344:346]
    pos15 = observation[347:349]
    direction = pos1 - pos15
    action = model.step(step, q=q, direction=direction)
    return action


def step_func(observation, **kwargs):
    com = observation[341:343]  # 2D center of mass
    return com


def done_func(index, result, **kwargs):
    """
    result: list of com in 1 simulation
    return: displacement in this simulation
    """
    displacement = np.linalg.norm(np.array(result[-1]) - np.array(result[0]), ord=2)
    d = np.linalg.norm(np.array(result[-1]) - np.array(result[-250]), ord=2)
    print('Trial {}: com displacement {:.2f}, last 250 steps {:.2f}'.format(index + 1, displacement, d))
    return displacement


if __name__ == '__main__':
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range='-40 40', max_episode_steps=max_episode_steps, reset_noise_scale=0.,
        density=1.2, viscosity=0.1, condim=3, friction='0.03 0.3'
    )
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space)
    print(env.observation_space)
    # model = SinusoidalMuscle(dt=env.dt, n=25, a=0.02, freq=0.8, psi=0.05)
    # model = ForwardMuscle(dt=env.dt, n=25, a=30 * np.pi / 180, freq=0.8, psi=0.05, kp=1, kv=0)
    model = ForwardPIDMuscle(
        dt=env.dt, n=25, a=30 * np.pi / 180, freq=0.8, psi=0.07,
        kp=1, kd=np.array([0.1 + i * 0.002 for i in range(24)]), kp_direction=0.15
    )
    results = simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False)
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
