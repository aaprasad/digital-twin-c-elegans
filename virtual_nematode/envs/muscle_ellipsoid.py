import gym
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.muscle_ellipsoid_v0 import swimmer
from gym_worm.envs.mujoco.position import position
from gym_worm.envs.mujoco.tendon import tendon
from gym_worm.wrappers.recorder import Recorder
from gym_worm.wrappers.sensor_observation import SensorObservation


def make_swimmer(n_bodies, joint_range, y_joint_ranges, theta, max_episode_steps, reset_noise_scale, density, viscosity, condim, friction):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, y_joint_ranges, theta, density, viscosity, condim, friction)
    xml_str = tendon(xml_str)
    xml_str = position(xml_str)
    xml_str = camera(xml_str)
    # with open('swimmer.xml', 'w') as f:
    #     f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = Recorder(env, camera_name=None)
    return env
