import gym
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.swimmer_v3_v1 import swimmer
from gym_worm.envs.mujoco.position import position
from gym_worm.envs.mujoco.tracking import tracking
from gym_worm.wrappers.distribution_observation import DistributionObservation
from gym_worm.wrappers.sensor_observation import SensorObservation
import numpy as np
from virtual_nematode.models.computational_model import ComputationalModelChemotaxis
from virtual_nematode.simulation import simulate


def fick(target, source, sigma=5):
    r = np.linalg.norm(target - source)
    c = np.exp(-r ** 2 / (2 * sigma ** 2))
    return c


def position_func(observation):
    """ get 2D center of mass and position """
    com = observation[56:58]
    position = observation[59:61]
    return com, position


def source_func(observation):
    """ get the position of the source """
    return observation[62:64]


def make_swimmer(x, y, n_bodies=25, joint_range='-100 100', body_len=0.1, rgba='0 1 0 1', max_episode_steps=1000):
    """ create swimmer env """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str = position(xml_str)
    xml_str = tracking(xml_str, x, y, rgba, kp=1)
    xml_str = camera(xml_str)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=0.)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = DistributionObservation(env, dt=env.dt, f=fick, position_func=position_func, source=None, source_func=source_func)
    return env


def action_func(model, step, observation, **kwargs):
    q = observation[3:27]
    q_vel = observation[32:56]
    g_p = observation[67]
    g_w = observation[68]
    action = model.step(step, q, q_vel, g_p, g_w)
    if step < 1250:
        theta = step / 100.  # anticlockwise
    else:
        theta = (2500 - step) / 100.  # clockwise
    radius = 10
    action = np.concatenate((action, [np.cos(theta) * radius, np.sin(theta) * radius]), axis=0)
    return action


def step_func(observation, **kwargs):
    concentration = observation[65]
    return concentration


def done_func(index, result, **kwargs):
    chemotaxis_index = np.mean(result)
    print('Trial {}: chemotaxis index {:.2f}'.format(index + 1, chemotaxis_index))
    return chemotaxis_index


if __name__ == '__main__':
    max_episode_steps = 2500
    env = make_swimmer(x=0, y=0, max_episode_steps=max_episode_steps)
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space)
    print(env.observation_space)
    kwargs = {'backward': False, 'omega': False, 'weathervane': True, 'random_walk': False}
    model = ComputationalModelChemotaxis(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2., **kwargs)
    results = simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False)
    print('{} trials: chemotaxis index mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
