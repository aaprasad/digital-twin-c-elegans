""" muscle worm: forward locomotion
joint range [-40, 40] degrees -> [-0.7, 0.7] rad
"""

from gym_worm.wrappers.muscle_action import JointAction
from virtual_nematode.envs.muscle_worm import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation.forward import simulate


def model_kwargs_func(observation, **kwargs):
    return {'q': observation[1:25], 'q_vel': observation[28:52]}


if __name__ == '__main__':
    max_episode_steps = 2500
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=0.7)
    env = JointAction(env)  # ctrl: joint -> muscle
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=1.5)
    simulate(env, model, model_kwargs_func, seed=None, max_episode_steps=max_episode_steps, trials=100, render=False)
