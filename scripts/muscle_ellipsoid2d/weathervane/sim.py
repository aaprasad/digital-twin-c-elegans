import numpy as np
from virtual_nematode.envs.muscle_ellipsoid2d import make_chemotaxis_swimmers
from virtual_nematode.models.forward import WeathervaneMuscle
from virtual_nematode.simulation import simulate


def position_func(observation, **kwargs):
    """ 2D center of mass and position of the first body segment """
    com, position = observation[341:343], observation[344:346]
    return com, position


def action_func(model, step, observation, **kwargs):
    """
    model: mathematical model of controller
    return: action
    """
    q = observation[4:28]
    q_vel = observation[32:56]
    g_p, g_w = observation[349], observation[350]
    action = model.step(step, q, q_vel, g_w)
    return action


def step_func(observation, **kwargs):
    """ accumulate concentrations along the path """
    concentration = [observation[347]]  # (1, )
    return concentration


def done_func(result, index=None, **kwargs):
    """ calculate chemotaxis index with concentrations along the path """
    result = np.array(result)
    chemotaxis_index = np.mean(result)
    start_concentration = result[0][0]
    print('Trial {}: chemotaxis index {:.2f}, start concentration {:.2f}'.format(index + 1, chemotaxis_index, start_concentration))
    return result


if __name__ == '__main__':
    seed = 11
    trials = 1
    camera_name = None  # set camera_name = 'track' or 'fixedcam', to record video
    env = make_chemotaxis_swimmers(
        seed=seed, trials=1, distance=10, position_func=position_func, n_bodies=25, joint_range='-40 40',
        max_episode_steps=3500, reset_noise_scale=0., camera_name=camera_name, return_func=False
    )
    env = env[0]
    print(env.action_space, env.observation_space)
    print(env.source)
    model = WeathervaneMuscle(dt=env.dt, n=25, a=30 * np.pi / 180, freq=0.8, psi=0.05, kp=1, kv=0)
    result = simulate(env, model, action_func, step_func, done_func, seed, trials=trials, render=True)  # (batch_size, max_episode_steps, 1)
    result = np.array(result)
    print('{} trials: chemotaxis index mean {:.2f}, start concentration mean {:.2f}'.format(result.shape[0], result.mean(), result[:, 0].mean()))
