""" swimmer: forward locomotion """

from virtual_nematode.envs.swimmer import make_swimmer_friction
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation import simulate


def model_kwargs_func(observation, **kwargs):
    """
    mjcf without arena: the root of robot body is 'worldbody/body'
    mjcf with arena: the root of robot body is 'worldbody/body/body'
    its observation space changes as well
    """
    return {'q': observation[2:26], 'q_vel': observation[30:54]}


if __name__ == '__main__':
    max_episode_steps = 2500
    env = make_swimmer_friction(
        n_bodies=25, joint_range='-100 100', body_len=0.1, camera_pos='0 -6 6', camera_z=50, camera_name=None,
        max_episode_steps=max_episode_steps, reset_noise_scale=0.1
    )
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    simulate(env, model, model_kwargs_func, seed=None, max_episode_steps=max_episode_steps, trials=1, render=False)
