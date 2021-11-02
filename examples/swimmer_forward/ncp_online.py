""" swimmer: open-loop control of forward locomotion
online (active) test NCP network's forward locomotion control
"""

import os
from virtual_nematode.trainers.ncp import prepare_model
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.testers.forward import single_tester


def encode_func(data, **kwargs):
    data = data.unsqueeze(-1)  # add input feature's dimension: []->[1]
    return data


if __name__ == '__main__':
    model_folder = os.path.join('runs', '')
    model_name = 'fully_connected'
    seed = 42
    max_episode_steps = 2500
    reset_noise_scale = 0.
    record = False
    kwargs = {'units': 30, 'output_dim': 11, 'in_features': 1}
    record_kwargs = {'camera_name': 'track', 'video_name': model_folder} if record else {}
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale, **record_kwargs)
    model = prepare_model(model_name, model_path=os.path.join(model_folder, 'model.pt'), **kwargs)
    single_tester(env, model, encode_func, seed, max_episode_steps)
