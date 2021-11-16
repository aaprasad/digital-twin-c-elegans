from data import x_func as data_func
import gym
import os
from sim import step_func as x_func
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model


def fully_connected():
    """ results
    units = 100
        100 trials: com displacement mean 21.76 / 2500 steps
        1 trial: com displacement 21.60 / 2500 steps
    units = 72
        100 trials: com displacement mean 21.75 / 2500 steps
        1 trial: com displacement 21.40 / 2500 steps
    """
    model_name = 'fully_connected'
    model = prepare_model(
        model_name, model_path=os.path.join(model_folder, 'model.pt'),
        **{'units': 72, 'output_dim': 24, 'in_features': 49}
    )
    return model, model_name


def ncp():
    model_name = 'ncp'
    model = prepare_model(
        model_name, model_path=os.path.join(model_folder, 'model.pt'),
        **{
            'in_features': 49, 'inter_neurons': 12, 'command_neurons': 36, 'motor_neurons': 24, 'sensory_fanout': 6,
            'inter_fanout': 12, 'recurrent_command_synapses': 6, 'motor_fanin': 6
        }
    )
    return model, model_name


if __name__ == '__main__':
    runs_folder = ''
    model_folder = os.path.join('runs', runs_folder)
    seed = 42
    reset_noise_scale = 1.745
    max_episode_steps = 2500
    env = make_swimmer(
        n_bodies=25, joint_range='-100 100', body_len=0.1, max_episode_steps=max_episode_steps,
        reset_noise_scale=reset_noise_scale
    )
    model, model_name = fully_connected()
    # model, model_name = ncp()
    tester(env, model, data_func, x_func, seed, max_episode_steps, model_folder, model_name, data_size=100)
    env = gym.wrappers.Monitor(env, directory=os.path.join('video', runs_folder), force=True)
    single_tester(env, model, data_func, x_func, seed, max_episode_steps)
