""" swimmer: chemotaxis based on weathervane mechanism
qpos[0:2]: x and y position of the front tip
observation[0:52]: qpos[2:27] and qvel[0:27]
    [0]: angle of the front tip (angle, rad)
    [1:25]: angle of the rotors (angle, rad)
    [25:27]: velocity of the tip along the x- and y-axis (velocity, m/s)
    [27]: angular velocity of front tip (angular velocity, rad/s)
    [28:52]: angular velocity of the rotors (angular velocity, rad/s)
observation[52:62]: additional
    [52:55]: 3D center of mass
    [55:58]: 3D position of the front tip
    [58:62]: concentration, g, g_p, g_w
"""
import gym
import numpy as np
from virtual_nematode.envs.swimmer import make_chemotaxis_swimmers
from virtual_nematode.models.computational_model import ComputationalModelChemotaxisVector
from virtual_nematode.simulation import simulate_vector


def position_func(observation, **kwargs):
    """ 2D center of mass and position of the first body segment """
    com, position = observation[52:54], observation[55:57]
    return com, position


def action_func(model, step, observation, vectorized=False, **kwargs):
    """
    model: mathematical model of controller
    return: action
    """
    observation = np.array(observation)
    if vectorized is False:  # action: (action_size, )
        q = observation[1:25]
        q_vel = observation[28:52]
        g_p, g_w = observation[60], observation[61]
    else:  # action: (batch_size, action_size)
        q = observation[:, 1:25]
        q_vel = observation[:, 28:52]
        g_p, g_w = observation[:, 60], observation[:, 61]
    action = model.step(step, q, q_vel, g_p, g_w)
    return action


def step_func(observation, vectorized=False, **kwargs):
    """ accumulate concentrations along the path """
    if vectorized is False:
        concentration = [observation[58]]  # (1, )
    else:
        concentration = np.expand_dims(observation[:, 58], axis=1)  # (batch_size, 1)
    return concentration


def done_func(result, index=None, vectorized=False, **kwargs):
    """ calculate chemotaxis index with concentrations along the path """
    result = np.array(result)
    if vectorized is False:  # result: (max_episode_steps, 1)
        chemotaxis_index = np.mean(result)
        start_concentration = result[0][0]
        print('Trial {}: chemotaxis index {:.2f}, start concentration {:.2f}'.format(index + 1, chemotaxis_index, start_concentration))
    else:
        result = np.swapaxes(result, 0, 1)  # (max_episode_steps, batch_size, 1) -> (batch_size, max_episode_steps, 1)
        for i in range(result.shape[0]):
            chemotaxis_index = np.mean(result[i])
            start_concentration = result[i, 0, 0]
            print('Trial {}: chemotaxis index {:.2f}, start concentration {:.2f}'.format(i + 1, chemotaxis_index, start_concentration))
    return result


if __name__ == '__main__':
    """ results
    100 trials: chemotaxis index mean 0.42 / 2500 steps
    100 trials: chemotaxis index mean 0.55, start concentration mean 0.01 / 3500 steps
    """
    # set trials = 1, camera_name = 'track' or 'fixedcam', to record video
    trails = 100
    data_size_per_trial = 1
    camera_name = None
    env = make_chemotaxis_swimmers(
        seed=11, trials=trails, distance=15, position_func=position_func, n_bodies=25, joint_range='-100 100', body_len=0.1,
        max_episode_steps=3500, reset_noise_scale=1.745, camera_name=camera_name, return_func=True
    )
    env = env * data_size_per_trial
    env = gym.vector.AsyncVectorEnv(env)
    print(env.action_space, env.observation_space)
    # print(env.single_action_space, env.single_observation_space)
    print([source.tolist() for source in env.get_attr('source')])
    kwargs = {'backward': False, 'omega': False, 'weathervane': True, 'random_walk': False, 'weathervane_reverse': False}
    model = ComputationalModelChemotaxisVector(
        dt=env.get_attr('dt')[0], seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2., n_bias=25,
        batch_size=env.num_envs, **kwargs
    )
    result = simulate_vector(env, model, action_func, step_func, done_func, seed=11, render=False)  # (batch_size, max_episode_steps, 1)
    print('{} trials: chemotaxis index mean {:.2f}, start concentration mean {:.2f}'.format(result.shape[0], result.mean(), result[:, 0].mean()))
