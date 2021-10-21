import torch
from virtual_nematode.testers.swimmer_forward import tester


def encode_func(data, observation):
    """ input_size: 1 + 11 + 11 """
    data = torch.tensor([data] + observation[1:12].tolist() + observation[15:].tolist(), dtype=torch.float64)
    return data


if __name__ == '__main__':
    model_folder = None
    model_name = 'fully_connected'
    mode = 'sine_wave'
    seed = 42
    data_size = 100
    reset_noise_scale = 1.745
    max_episode_steps = 2500
    kwargs = {'units': 50, 'output_dim': 11, 'in_features': 23}
    tester(encode_func, seed, max_episode_steps, reset_noise_scale, model_folder, model_name, data_size, mode, **kwargs)
