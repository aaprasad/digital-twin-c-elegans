""" swimmer: forward locomotion
swimmer configuration
    n_bodies = 25
    receive muscle ctrl signal
    perform joint ctrl
"""

from gym_worm.wrappers.muscle_action import MuscleAction, JointAction
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation.forward import simulate


def model_kwargs_func(observation, **kwargs):
    return {'q': observation[1:25], 'q_vel': observation[28:52]}


if __name__ == '__main__':
    """ results
    100 trials: com displacement mean 16.89 / 2500 steps
    """
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range='-100 100', body_len=0.1, camera_pos='0 -6 6', camera_z=50, camera_name=None,
        max_episode_steps=max_episode_steps, reset_noise_scale=1.745
    )
    env = MuscleAction(env)  # ctrl: muscle -> joint
    env = JointAction(env)  # ctrl: joint -> muscle
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    simulate(env, model, model_kwargs_func, seed=None, max_episode_steps=max_episode_steps, trials=100)
