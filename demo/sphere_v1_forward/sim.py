import gym
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.position import position
from gym_worm.envs.mujoco.sphere_v0 import swimmer
from gym_worm.wrappers.recorder import Recorder
from gym_worm.wrappers.sensor_observation import SensorObservation
import numpy as np
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation import simulate


def make_swimmer(n_bodies=25, joint_range='-100 100', max_episode_steps=1000, reset_noise_scale=0.1):
    """ create swimmer env """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range)
    xml_str = position(xml_str)
    xml_str = camera(xml_str)
    env = gym.make(
        'Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env


def action_func(model, step, observation, **kwargs):
    """
    model: mathematical model of controller
    return: action
    """
    q = observation[3:27]
    q_vel = observation[30:54]
    action = model.step(step, q, q_vel)
    return action


def step_func(observation, **kwargs):
    """
    return: 2D center of mass
    """
    com = observation[54:56]
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
    # gym.logger.set_level(gym.logger.DEBUG)
    max_episode_steps = 2500
    env = make_swimmer(n_bodies=25, joint_range='-100 100', max_episode_steps=max_episode_steps, reset_noise_scale=1.745)
    # env = gym.wrappers.Monitor(env, directory='video/sphere_v1', force=True)
    print(env.action_space)
    print(env.observation_space)
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    results = simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False)
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
