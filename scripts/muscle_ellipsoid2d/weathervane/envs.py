import gymnasium as gym
from gym_worm.envs.mujoco.chemotaxis import chemotaxis
from gym_worm.envs.mujoco.camera import camera
from gym_worm.envs.mujoco.forage import forage
from gym_worm.envs.mujoco.muscle_ellipsoid2d_v0 import swimmer
from gym_worm.envs.mujoco.position import position
from gym_worm.envs.mujoco.t_maze import t_maze
from gym_worm.wrappers.distribution_observation import DistributionObservation
from gym_worm.wrappers.recorder import Recorder
from gym_worm.wrappers.sensor_observation import SensorObservation
from sim import position_func
from virtual_nematode.envs.swimmer import fick


def make_swimmer(
    n_bodies, joint_range, max_episode_steps, reset_noise_scale, pos, source, position_func, density,
    viscosity, condim, friction, cone, distribution_func=fick
):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction, cone)
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    with open('swimmer.xml', 'w') as f:
        f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v1', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale, position=pos
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = DistributionObservation(env, dt=env.dt, f=distribution_func, source=source, position_func=position_func)
    env = Recorder(env, camera_name=None)
    return env


def make_swimmer_weathervane_fixed(
    n_bodies, joint_range, max_episode_steps, reset_noise_scale, pos, source, position_func, density,
    viscosity, condim, friction, cone, distribution_func=fick
):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction, cone)
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    xml_str = chemotaxis(xml_str, x=source[0], y=source[1])
    with open('swimmer_chemotaxis.xml', 'w') as f:
        f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v1', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale, position=pos
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = DistributionObservation(env, dt=env.dt, f=distribution_func, source=source, position_func=position_func)
    env = Recorder(env, camera_name=None)
    return env


def make_swimmer_forage_fixed(
    n_bodies, joint_range, max_episode_steps, reset_noise_scale, pos, source, position_func, density,
    viscosity, condim, friction, cone, distribution_func=fick
):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction, cone)
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    xml_str = forage(xml_str, seed=7)
    with open('swimmer_forage.xml', 'w') as f:
        f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v1', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale, position=pos
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = DistributionObservation(env, dt=env.dt, f=distribution_func, source=source, position_func=position_func)
    env = Recorder(env, camera_name=None)
    return env


def make_swimmer_maze_fixed(
    n_bodies, joint_range, max_episode_steps, reset_noise_scale, pos, source, position_func, density,
    viscosity, condim, friction, cone, distribution_func=fick
):
    xml_str = swimmer('swimmer.xml', n_bodies, joint_range, density, viscosity, condim, friction, cone)
    xml_str = position(xml_str)
    xml_str = camera(xml_str, camera_pos='-1.25 0 5', camera_xyaxes='1 0 0 0 1 0')
    xml_str = t_maze(xml_str, mesh_file='t_maze.stl', height=0)
    with open('swimmer_maze.xml', 'w') as f:
        f.write(xml_str.decode('utf-8'))
    env = gym.make(
        'Swimmer-v3-v1', xml_str=xml_str.decode('utf-8'), exclude_current_positions_from_observation=False,
        reset_noise_scale=reset_noise_scale, position=pos
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.ClipAction(env)
    env = SensorObservation(env)
    env = DistributionObservation(env, dt=env.dt, f=distribution_func, source=source, position_func=position_func)
    env = Recorder(env, camera_name=None)
    return env


if __name__ == '__main__':
    # env = make_swimmer(
    #     n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
    #     max_episode_steps=2500, reset_noise_scale=0.6,
    #     pos=(5, 0), source=(0, 0), position_func=position_func,
    #     density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    # )
    env = make_swimmer_weathervane_fixed(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=2500, reset_noise_scale=0.6,
        pos=(0, 0), source=(3, 4), position_func=position_func,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    # env = make_swimmer_forage_fixed(
    #     n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
    #     max_episode_steps=2500, reset_noise_scale=0.6,
    #     pos=(5, 0), source=(0, 0), position_func=position_func,
    #     density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    # )
    # env = make_swimmer_maze_fixed(
    #     n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
    #     max_episode_steps=2500, reset_noise_scale=0.6,
    #     pos=(5, 0), source=(0, 0), position_func=position_func,
    #     density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    # )
