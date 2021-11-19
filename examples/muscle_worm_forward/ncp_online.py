from data import x_func as data_func
import gym
import os
from sim import step_func as x_func
import torch
from virtual_nematode.envs.muscle_worm import make_swimmer
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model


def fully_connected():
    """ results
    units = 200, batch_size = 256
        100 trials: com displacement mean 3.25 / 2500 steps
        1 trial: com displacement 3.42 / 2500 steps
    units = 150, batch_size = 512
        100 trials: com displacement mean 2.78 / 2500 steps
        1 trial: com displacement 2.95 / 2500 steps
    units = 100, batch_size = 1024
        100 trials: com displacement mean 2.38 / 2500 steps
        1 trial: com displacement 2.60 / 2500 steps
    units = 128, batch_size = 512
        100 trials: com displacement mean 3.02 / 2500 steps
        1 trial: com displacement 3.23 / 2500 steps
    units = 150, batch_size = 512, output_mapping='sigmoid'
        100 trials: com displacement mean 3.37 / 2500 steps
        1 trial: com displacement 3.95 / 2500 steps
    units = 150, batch_size = 512, output_mapping='relu'
        100 trials: com displacement mean 3.47 / 2500 steps
        1 trial: com displacement 3.66 / 2500 steps
    """
    model_name = 'fully_connected'
    model = prepare_model(
        model_name, model_path=os.path.join(model_folder, 'model.pt'),
        **{'units': 100, 'output_dim': 96, 'in_features': 193, 'output_mapping': 'linear-relu'}
    )
    return model, model_name


if __name__ == '__main__':
    runs_folder = ''
    model_folder = os.path.join('runs', runs_folder)
    seed = 42
    reset_noise_scale = 0.7
    max_episode_steps = 2500
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale)
    model, model_name = fully_connected()
    tester(env, model, data_func, x_func, seed, max_episode_steps, model_folder, model_name, data_size=100)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder), force=True)
    _, y = single_tester(env, model, data_func, x_func, seed, max_episode_steps)
    torch.save(y, 'data/{}.pt'.format(runs_folder))  # action sequence
