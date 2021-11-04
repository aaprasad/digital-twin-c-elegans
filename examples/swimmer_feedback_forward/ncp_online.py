from data import x_func
import os
from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.testers.forward import tester, single_tester
from virtual_nematode.trainers.ncp import prepare_model


if __name__ == '__main__':
    """ results
    100 trials: com displacement mean 40.76 / 2500 steps
    1 trial: com displacement 40.44 / 2500 steps
    """
    model_folder = os.path.join('runs', '')
    model_name = 'fully_connected'
    seed = 42
    data_size = 100
    reset_noise_scale = 1.745
    max_episode_steps = 2500
    record = False
    kwargs = {'units': 50, 'output_dim': 11, 'in_features': 23}
    record_kwargs = {'camera_name': 'track', 'video_name': model_folder} if record else {}
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale, **record_kwargs)
    model = prepare_model(model_name, model_path=os.path.join(model_folder, 'model.pt'), **kwargs)
    tester(env, model, x_func, seed, max_episode_steps, model_folder, model_name, data_size)
    single_tester(env, model, x_func, seed, max_episode_steps)
