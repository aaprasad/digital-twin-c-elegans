from data import x_func
import gym
from gym_worm.wrappers.muscle_action import MuscleAction
import os
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model


if __name__ == '__main__':
    """ results
    100 trials: com displacement mean 21.72 / 2500 steps
    1 trial: com displacement 21.46 / 2500 steps
    """
    runs_folder = ''
    model_folder = os.path.join('runs', runs_folder)
    model_name = 'fully_connected'
    seed = 42
    reset_noise_scale = 1.745
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=max_episode_steps,
        reset_noise_scale=reset_noise_scale
    )
    env = MuscleAction(env)
    model = prepare_model(
        model_name, model_path=os.path.join(model_folder, 'model.pt'),
        **{'units': 100, 'output_dim': 96, 'in_features': 49}
    )
    tester(env, model, x_func, seed, max_episode_steps, model_folder, model_name, data_size=100)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder), force=True)
    single_tester(env, model, x_func, seed, max_episode_steps)
