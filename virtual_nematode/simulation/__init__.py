# from matplotlib import pyplot as plt
import numpy as np
# import seaborn as sns


def simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False):
    """ forward sinusoidal movement
    action_func: called before every step, generate action
    step_func: called after every step, process observation and collect result of 1 simulation
    done_func: called after every simulation, process simulation result and collect results of all simulations
    **kwargs: configure mathematical model
    """
    results = []  # results of all simulations
    # angles, actions = [], []
    for i in range(trials):
        result = []  # results of 1 simulation
        env.seed(seed=seed + i if seed is not None else None)
        observation = env.reset()
        model.reset()
        for step in range(10 ** 6):
            if render is True:
                env.render()
            action = action_func(model=model, step=step, observation=observation)
            observation, reward, done, info = env.step(action)
            # angles.append(observation[4:28])
            # actions.append(action)
            result.append(step_func(observation=observation))
            if done:
                results.append(done_func(index=i, result=result))
                break
    # plt.subplot(1, 2, 1)
    # angles = np.array(angles)
    # angles_max = np.max(np.abs(angles))
    # sns.heatmap(angles.T, cmap='coolwarm', vmin=-angles_max, vmax=angles_max)
    # plt.subplot(1, 2, 2)
    # actions = np.array(actions)
    # sns.heatmap(actions.clip(0, 1).T, cmap='coolwarm')
    # plt.show()
    return results


def simulate_vector(env, model, action_func, step_func, done_func, seed=None, render=False):
    env.seed(seed)
    observation = env.reset()
    result = []
    for step in range(10 ** 6):
        if render is True:
            env.render()
        action = action_func(model=model, step=step, observation=observation, vectorized=True)
        observation, reward, done, info = env.step(action)
        result.append(step_func(observation=observation, action=action, vectorized=True))
        if done.all():
            results = done_func(result=result, vectorized=True)
            break
    return results
