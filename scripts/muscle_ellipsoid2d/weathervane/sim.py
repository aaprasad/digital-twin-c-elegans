"""
action space: Box(0.0, 1.0, (95,), float32)
observation space: Box(-inf, inf, (62,), float64)
    [0:56]: Ellipsoid2d-v0 observation space
    [56:59]: x-, y- and z-coordinates of the robot's center of mass (length, m)
    [59:62]: x-, y- and z-coordinates of the front tip (length, m)
    [62:66]: distribution related observation
"""

import numpy as np
from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer_weathervane
from virtual_nematode.models.muscle import WeathervanePIDMuscle
from virtual_nematode.simulation import simulate


def position_func(observation, **kwargs):
    """ 2D center of mass and position of the first body segment """
    com, position = observation[56:58], observation[59:61]
    return com, position


def action_func(model, step, observation, **kwargs):
    """
    model: mathematical model of controller
    return: action
    """
    q = observation[4:28]
    # q_vel = observation[32:56]
    # g_p = observation[64]
    g_w = observation[65]
    action = model.step(step, q=q, g_w=g_w)
    return action


def step_func(observation, **kwargs):
    """ record stats """
    com, position = observation[56:58], observation[59:61]
    gradient = observation[62:66]  # c, g, g_p, g_w
    return np.concatenate((com, position, gradient))


def done_func(result, index=None, **kwargs):
    """ calculate chemotaxis index with concentrations along the path """
    result = np.array(result)
    chemotaxis_index = np.mean(result[:, 4])
    start_concentration = result[0, 4]
    print('Trial {}: chemotaxis index {:.2f}, start concentration {:.2f}'.format(index + 1, chemotaxis_index, start_concentration))
    return result


if __name__ == '__main__':
    """ 100 trials
    k_w = 1, seed = 3: chemotaxis index mean 0.66, start concentration mean 0.37
    k_w = 0, seed = 3: chemotaxis index mean 0.14, start concentration mean 0.37
    """
    seed = 3  # None
    env = make_swimmer_weathervane(
        n_bodies=25, joint_range='-90 90', max_episode_steps=2500, reset_noise_scale=0.6, distance=15,
        position_func=position_func, density=1.2, viscosity=0.1, condim=3, friction='1 1', source=(0, 0)
    )  # distance = 3 * sigma
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space, env.observation_space)
    print(env.source)
    model = WeathervanePIDMuscle(
        k_w=1, dt=env.dt, n=25, a=0.6, freq=0.8, psi=0.07,
        kp=np.concatenate(([1 + i * 0.2 for i in range(12)], [3.2 - i * 0.2 for i in range(12)])),
        kd=0.15
    )
    result = simulate(env, model, action_func, step_func, done_func, seed, trials=1, render=False)  # (batch_size, max_episode_steps, 1)
    result = np.array(result)
    print('{} trials: chemotaxis index mean {:.2f}, start concentration mean {:.2f}'.format(result.shape[0], result[:, :, 4].mean(), result[:, 0, 4].mean()))
