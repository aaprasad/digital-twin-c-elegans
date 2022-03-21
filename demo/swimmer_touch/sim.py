import gym
import numpy as np
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.maze import maze
from gym_worm.envs.mujoco.swimmer_v3_v1 import swimmer
from gym_worm.envs.mujoco.touch import touch
from gym_worm.wrappers.sensor_observation import SensorObservation
from virtual_nematode.models.tap import Tap
from virtual_nematode.simulation import simulate


def make_swimmer(n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=1000, reset_noise_scale=0.1):
    """ create swimmer env """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str = touch(xml_str, body_len)
    xml_str = maze(xml_str, width=3.5, wall_pos=[], wall_size=[])
    xml_str = camera(xml_str)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = SensorObservation(env)
    return env


def action_func(model, step, observation, **kwargs):
    q = observation[1:25]
    q_vel = observation[28:52]
    force = step_func(observation)
    action = model.step(step, q, q_vel, force)
    return action


def step_func(observation, **kwargs):
    """ get touch force """
    force = observation[52:54]
    return force


def done_func(index, result, **kwargs):
    return np.array(result)


if __name__ == '__main__':
    max_episode_steps = 2500
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=1.745)
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space)
    print(env.observation_space)
    model = Tap(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    results = simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False)
    for i, r in enumerate(results[0]):
        if r.any():  # touch force
            print('step {}: {}'.format(i, r))
