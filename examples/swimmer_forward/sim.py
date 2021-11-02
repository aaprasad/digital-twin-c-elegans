""" swimmer: forward locomotion """

from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation.forward import simulate


def model_kwargs_func(observation, **kwargs):
    return {'q': observation[1:12], 'q_vel': observation[15:]}


if __name__ == '__main__':
    max_episode_steps = 2500
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=0.1, camera_z=50, camera_name=None)
    model = Forward(dt=env.dt, seed=None)
    simulate(env, model, model_kwargs_func, seed=None, max_episode_steps=max_episode_steps, trials=1)
