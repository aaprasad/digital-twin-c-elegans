""" swimmer: forward locomotion """

import numpy as np
from virtual_nematode.envs.swimmer_v3_v2 import make_swimmer
from virtual_nematode.models.forward import Forward


def simulate(seed=None, max_episode_steps=2500, reset_noise_scale=0.1, trials=1):
    """ forward sinusoidal movement """
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale, camera_z=50, camera_name=None)
    displacements = []
    if trials > 1:
        seed = None  # ensure that seed is different in each trial
    for i in range(trials):
        env.seed(seed)
        observation = env.reset()
        model = Forward(dt=env.dt, seed=seed)
        for step in range(10 ** 6):
            # env.render()
            action = model.step(step=step, q=observation[1:12], q_vel=observation[15:])
            observation, reward, done, info = env.step(action)
            if done:
                d = np.linalg.norm(np.array(env.stats['com'][-1]) - np.array(env.stats['com'][0]), ord=2)
                displacements.append(d)
                print('Trial {}: com displacement {:.2f} / {} steps'.format(i + 1, d, step + 1))
                break
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(displacements), np.mean(displacements), max_episode_steps))


if __name__ == '__main__':
    """ results
    100 trials: com displacement mean 40.84 / 2500 steps
    """
    simulate(max_episode_steps=2500, reset_noise_scale=1.745, seed=None, trials=100)
