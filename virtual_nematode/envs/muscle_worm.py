import gym
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.muscle_worm_v0 import swimmer
from gym_worm.wrappers.muscle_observation import MuscleObservation
from gym_worm.wrappers.position import Position


def make_swimmer(
    n_bodies=25, joint_range='-40 40', body_len=0.1, muscle_len=0.1, y_sidesite=0.1, z_medial=0.02, z_lateral=0.04,
    arrangement=None, camera_pos='0 -6 6', camera_z=50, max_episode_steps=1000, reset_noise_scale=0.1
):
    """ create swimmer env """
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, body_len, muscle_len, y_sidesite, z_medial, z_lateral, arrangement)
    xml_str = camera(xml_str, camera_pos, camera_z)
    env = gym.make('Swimmer-v3-v0', xml_str=xml_str.decode('utf-8'), reset_noise_scale=reset_noise_scale)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = MuscleObservation(env)
    env = Position(env)
    return env
