""" swimmer: forward locomotion
swimmer configuration
    joint size = 0.1 <- radius = 0.04mm
    whole body len = 2.5 <- length = 1mm
    joint amount = 24 <- amount of muscles in one quadrant = 24
    n_bodies = 25
    body len = 0.1
observation space: (52,) including qpos[2:27], qvel[0:25],
    qpos[0:27]: x_pos, y_pos, ?, q[0:24]
    qvel[0:27]: x_vel, y_vel, ?, q_vel[0:24]
"""

from virtual_nematode.envs.swimmer_v3_v2 import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation.forward import simulate


def model_kwargs_func(observation):
    return {'q': observation[1:25], 'q_vel': observation[28:]}


if __name__ == '__main__':
    n_bodies = 25
    joint_range = '-100 100'
    body_len = 0.1
    camera_pos = '0 -6 6'
    camera_z = 50
    camera_name = None
    max_episode_steps = 2500
    video_name = 'swimmer'
    reset_noise_scale = 0.1
    env = make_swimmer(
        n_bodies, joint_range, body_len, camera_pos, camera_z, camera_name, max_episode_steps, video_name,
        reset_noise_scale
    )
    # print(env.action_space)
    # print(env.observation_space)
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    simulate(env, model, model_kwargs_func, seed=None, max_episode_steps=max_episode_steps, trials=1)
