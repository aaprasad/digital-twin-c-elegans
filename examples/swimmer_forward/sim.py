""" swimmer: forward locomotion """

import numpy as np
from virtual_nematode.envs.swimmer_v3_v2 import make_swimmer
from virtual_nematode.models.forward import Forward


def simulate(seed=None, max_episode_steps=2500, reset_noise_scale=0.1):
    """ forward sinusoidal movement """
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale, camera_z=50, camera_name=None)
    env.seed(seed)
    observation = env.reset()
    model = Forward(dt=env.dt, seed=seed)
    for step in range(10 ** 6):
        # env.render()
        action = model.step(step=step, q=observation[1:12], q_vel=observation[15:])
        observation, reward, done, info = env.step(action)
        if done:
            break
    displacement = np.linalg.norm(np.array(env.stats['com'][-1]) - np.array(env.stats['com'][0]), ord=2)
    print('com displacement {:.2f} / {} steps'.format(displacement, max_episode_steps))


if __name__ == '__main__':
    simulate(max_episode_steps=2500, seed=None)
