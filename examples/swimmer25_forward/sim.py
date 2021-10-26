""" swimmer: forward locomotion """

from virtual_nematode.envs.swimmer_v3_v2 import make_swimmer
from virtual_nematode.simulation.forward import simulate


if __name__ == '__main__':
    """ swimmer configuration
    joint size = 0.1 <- radius = 0.04mm
    whole body len = 2.5 <- length = 1mm
    joint amount = 24 <- amount of muscles in one quadrant = 24
    body len = 0.1
    """
    n_bodies = 25
    joint_range = '-100 100'
    body_len = 0.1
    camera_pos = '0 -6 6'
    camera_z = 50
    camera_name = None
    max_episode_steps = 2500
    video_name = 'swimmer_forward'
    reset_noise_scale = 0.1
    env = make_swimmer(
        n_bodies, joint_range, body_len, camera_pos, camera_z, camera_name, max_episode_steps, video_name,
        reset_noise_scale
    )
    simulate(env, seed=None, max_episode_steps=max_episode_steps, trials=1)
