import gym
import os
import torch
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model


def encode_func(stimuli, observation, **kwargs):
    data = torch.tensor([stimuli] + observation[1:25].tolist() + observation[28:52].tolist(), dtype=torch.float64)
    return data


if __name__ == '__main__':
    """ results
    100 trials: com displacement mean 14.27 / 2500 steps
    1 trial: com displacement 13.56 / 2500 steps
    """
    model_folder = os.path.join('runs', '')
    model_name = 'fully_connected'
    seed = 42
    reset_noise_scale = 1.745
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=max_episode_steps,
        reset_noise_scale=reset_noise_scale
    )
    model = prepare_model(
        model_name, model_path=os.path.join(model_folder, 'model.pt'),
        **{'units': 100, 'output_dim': 24, 'in_features': 49}
    )
    tester(env, model, encode_func, seed, max_episode_steps, model_folder, model_name, data_size=100)
    env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    single_tester(env, model, encode_func, seed, max_episode_steps)
