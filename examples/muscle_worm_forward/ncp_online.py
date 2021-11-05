from data import x_func
import gym
import os
from virtual_nematode.envs.muscle_worm import make_swimmer
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model


if __name__ == '__main__':
    runs_folder = ''
    model_folder = os.path.join('runs', runs_folder)
    model_name = 'fully_connected'
    seed = 7
    reset_noise_scale = 0.7
    max_episode_steps = 2500
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale)
    model = prepare_model(
        model_name, model_path=os.path.join(model_folder, 'model.pt'),
        **{'units': 200, 'output_dim': 96, 'in_features': 193}
    )
    tester(env, model, x_func, seed, max_episode_steps, model_folder, model_name, data_size=100)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder), force=True)
    single_tester(env, model, x_func, seed, max_episode_steps)
