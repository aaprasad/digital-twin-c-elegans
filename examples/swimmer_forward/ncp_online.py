""" swimmer: open-loop control of forward locomotion
online (active) test NCP network's forward locomotion control
"""

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
    single_tester(encode_func, seed, max_episode_steps, reset_noise_scale, model_folder, model_name, mode, record, **kwargs)
