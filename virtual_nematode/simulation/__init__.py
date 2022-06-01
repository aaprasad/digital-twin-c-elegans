from matplotlib import pyplot as plt
import numpy as np


def simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False):
    """ forward sinusoidal movement
    action_func: called before every step, generate action
    step_func: called after every step, process observation and collect result of 1 simulation
    done_func: called after every simulation, process simulation result and collect results of all simulations
    **kwargs: configure mathematical model
    """
    results = []  # results of all simulations
    if trials > 1:
        seed = None  # ensure that seed is different in each trial
    # obs_list = [[], []]
    for i in range(trials):
        result = []  # results of 1 simulation
        env.seed(seed)
        observation = env.reset()
        model.reset()
        for step in range(10 ** 6):
            if render is True:
                env.render()
            action = action_func(model=model, step=step, observation=observation)
            observation, reward, done, info = env.step(action)
            # for j, index in enumerate([4, 27]):
            #     obs_list[j].append(observation[index] * 180 / np.pi)
            result.append(step_func(observation=observation))
            if done:
                results.append(done_func(index=i, result=result))
                break
    # for x, label in zip(obs_list, [4, 27]):
    #     plt.plot(x, label=str(label))
    # plt.savefig('observation.png')
    # plt.legend()
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
