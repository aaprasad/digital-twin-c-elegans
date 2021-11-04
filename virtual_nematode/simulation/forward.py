import numpy as np


def simulate(env, model, model_kwargs_func, seed=None, max_episode_steps=2500, trials=1, render=False):
    """ forward sinusoidal movement
    model_kwargs_func: function, take in observation and return the needed kwargs for model.step()
    **kwargs: configure mathematical model
    """
    displacements = []
    if trials > 1:
        seed = None  # ensure that seed is different in each trial
    for i in range(trials):
        coms = []
        env.seed(seed)
        observation = env.reset()
        for step in range(10 ** 6):
            if render is True:
                env.render()
            action = model.step(step=step, **model_kwargs_func(observation=observation))
            observation, reward, done, info = env.step(action)
            coms.append(info['com'])
            if done:
                d = np.linalg.norm(np.array(coms[-1]) - np.array(coms[0]), ord=2)
                displacements.append(d)
                print('Trial {}: com displacement {:.2f} / {} steps'.format(i + 1, d, step + 1))
                break
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(displacements), np.mean(displacements), max_episode_steps))
