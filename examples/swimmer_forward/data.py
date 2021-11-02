""" swimmer: open-loop control of forward locomotion
generate forward simulation dataset
"""

from sim import make_swimmer
from virtual_nematode.data.simulation import generate_dataset
from virtual_nematode.models.forward import Forward


def model_kwargs_func(observation, **kwargs):
    return {'q': observation[1:12], 'q_vel': observation[15:]}


def x_func(stimuli, **kwargs):
    return [stimuli]


if __name__ == '__main__':
    """
    mode: stimuli mode
    x: torch.Tensor, (max_episode_steps, input_size)
        action sequence of the first joint in trials
    y: torch.Tensor, (max_episode_steps, action_size)
        action sequences in trials
    """
    input_size = 1
    data_size = 1
    reset_noise_scale = 0.
    max_episode_steps = 5000
    seed = 42
    mode = 'sine_wave'  # or 'square_wave'
    save_name = 'data.pt'
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale)
    model = Forward(dt=env.dt, seed=seed)
    generate_dataset(env, model, input_size, model_kwargs_func, x_func, data_size, seed, max_episode_steps, mode, save_name)
