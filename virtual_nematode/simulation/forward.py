import numpy as np
from virtual_nematode.models.forward import Forward


def simulate(env, seed=None, max_episode_steps=2500, trials=1):
    """ forward sinusoidal movement """
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
