""" swimmer: open-loop control of forward locomotion
online (active) test NCP network's forward locomotion control
"""

from virtual_nematode.envs.swimmer_v3_v2 import make_swimmer
from virtual_nematode.testers.swimmer_forward import single_tester


def encode_func(data, **kwargs):
    data = data.unsqueeze(-1)  # add input feature's dimension: []->[1]
    return data


if __name__ == '__main__':
    model_folder = None
    model_name = 'fully_connected'
    mode = 'sine_wave'
    seed = 42
    max_episode_steps = 2500
    reset_noise_scale = 0.
    record = False
    kwargs = {'units': 30, 'output_dim': 11, 'in_features': 1}
    record_kwargs = {'camera_name': 'track', 'video_name': model_folder} if record else {}
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale, **record_kwargs)
    single_tester(env, encode_func, seed, max_episode_steps, model_folder, model_name, mode, **kwargs)
