import gymnasium as gym
from gym_worm.envs.mujoco.swimmer_v3_v1 import swimmer
from gym_worm.envs.mujoco.position import position
from gym_worm.envs.mujoco.position_actuator import position_actuator
from gym_worm.wrappers.sensor_observation import SensorObservation
from gym_worm.wrappers.skip_frame import SkipFrame
import numpy as np
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation import simulate


def make_swimmer(n_bodies, joint_range, body_len, max_episode_steps, reset_noise_scale):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len)
    xml_str = position_actuator(xml_str, joint_range, kp=1)
    xml_str = position(xml_str)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = SkipFrame(env, skip=10)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    return env


def action_func(model, step, observation, **kwargs):
    q = observation[1:25]
    q_vel = observation[28:52]
    action = model.step(step, q, q_vel)
    return action


def step_func(observation, **kwargs):
    com = observation[52:54]  # 2D center of mass
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
    max_episode_steps = 2500
    env = make_swimmer(n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=max_episode_steps, reset_noise_scale=1.745)
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space)
    print(env.observation_space)
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2., motor=False)  # q_max, psi, freq
    results = simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False)
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
