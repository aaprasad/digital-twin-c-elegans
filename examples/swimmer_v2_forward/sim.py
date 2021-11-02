""" swimmer: forward locomotion """

from gym_worm.wrappers.muscle_action import MuscleAction
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.simulation.forward import simulate


def model_kwargs_func(observation, **kwargs):
    return {'q': observation[1:25], 'q_vel': observation[28:]}


def action_func(env, action):
    action = env.reverse_action(action)
    return action


def check_wrapper(env):
    """ check if gym.ActionWrapper works """
    print(env.action_space)  # muscle ctrl: Box([0., ...], [1., ...], (96,), float32)
    print(env.unwrapped.action_space)  # joint ctrl: Box([-1., ...], [1., ...], (24,), float32)
    action = env.unwrapped.action_space.sample()
    action_reconstructed = env.action(env.reverse_action(action))  # ctrl: joint -> muscle -> joint
    print(action == action_reconstructed)  # reconstructed joint ctrl should be the same as the original one


if __name__ == '__main__':
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range='-100 100', body_len=0.1, camera_pos='0 -6 6', camera_z=50, camera_name=None,
        max_episode_steps=max_episode_steps, video_name='swimmer', reset_noise_scale=0.1
    )
    env = MuscleAction(env)
    # check_wrapper(env)
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2.)
    simulate(env, model, model_kwargs_func, action_func=action_func, seed=None, max_episode_steps=max_episode_steps, trials=1)
