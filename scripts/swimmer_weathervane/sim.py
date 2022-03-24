import numpy as np
from sim_vector import position_func, action_func, step_func, done_func
from virtual_nematode.envs.swimmer import make_chemotaxis_swimmers
from virtual_nematode.models.computational_model import ComputationalModelChemotaxis
from virtual_nematode.simulation import simulate


if __name__ == '__main__':
    trials = 1
    camera_name = None  # set camera_name = 'track' or 'fixedcam', to record video
    env = make_chemotaxis_swimmers(
        seed=11, trials=trials, distance=15, position_func=position_func, n_bodies=25, joint_range='-100 100', body_len=0.1,
        max_episode_steps=3500, reset_noise_scale=1.745, camera_name=camera_name, return_func=False
    )
    env = env[0]
    print(env.action_space, env.observation_space)
    print(env.source)
    kwargs = {'backward': False, 'omega': False, 'weathervane': True, 'random_walk': False, 'weathervane_reverse': False}
    model = ComputationalModelChemotaxis(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=2., n_bias=25, **kwargs)
    result = simulate(env, model, action_func, step_func, done_func, seed=11, trials=trials, render=False)  # (batch_size, max_episode_steps, 1)
    result = np.array(result)
    print('{} trials: chemotaxis index mean {:.2f}, start concentration mean {:.2f}'.format(result.shape[0], result.mean(), result[:, 0].mean()))
