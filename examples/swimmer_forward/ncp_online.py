""" swimmer: open-loop control of forward locomotion
online (active) test NCP network's forward locomotion control
"""

import numpy as np
import os
from virtual_nematode.models.forward import Forward
from virtual_nematode.testers.swimmer_forward import test_func
from virtual_nematode.trainers.ncp import prepare_model
from virtual_nematode.envs.swimmer_forward import make_swimmer
import torch


def encode_func(data, **kwargs):
    data = data.unsqueeze(-1)  # add input feature's dimension: []->[1]
    return data


def online_test_once(
    seed=42, max_episode_steps=2500, model_folder=None, model_name='fully_connected', mode='sine_wave', record=True,
    **kwargs
):
    """ online test and record video """
    np.random.seed(seed)
    torch.manual_seed(seed)
    assert model_folder is not None, 'model_folder can not be {}'.format(model_folder)
    if record is True:
        record_kwargs = {'camera_name': 'track', 'video_name': model_folder}
    else:
        record_kwargs = {}
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=0., **record_kwargs)
    model = prepare_model(model_name, model_path=os.path.join('runs', model_folder, 'model.pt'), **kwargs)
    math_model = Forward(dt=env.dt, seed=seed)
    x, _ = test_func(env, model, math_model, encode_func, mode)
    displacement = torch.linalg.norm(x[-1, :] - x[0, :], ord=2).item()
    print('com displacement {:.2f} / {} steps'.format(displacement, max_episode_steps))


if __name__ == '__main__':
    model_folder = None
    model_name = 'fully_connected'
    mode = 'sine_wave'
    kwargs = {'units': 30, 'output_dim': 11, 'in_features': 1}
    # tester(test_func, data_size=100, model_folder=model_folder, model_name=model_name, mode=mode, **kwargs)
    online_test_once(model_folder=model_folder, model_name=model_name, mode=mode, record=False, **kwargs)
