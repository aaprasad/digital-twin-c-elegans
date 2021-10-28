""" swimmer: forward locomotion """

from virtual_nematode.envs.swimmer_v3_v2 import make_swimmer
from virtual_nematode.simulation.forward import simulate


def model_step_kwargs_func(observation):
    return {'q': observation[1:12], 'q_vel': observation[15:]}


if __name__ == '__main__':
    """ results
    100 trials: com displacement mean 40.84 / 2500 steps
    """
    max_episode_steps = 2500
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=1.745, camera_z=50, camera_name=None)
    simulate(env, model_step_kwargs_func, seed=None, max_episode_steps=max_episode_steps, trials=100)
