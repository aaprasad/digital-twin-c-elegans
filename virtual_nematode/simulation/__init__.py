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
    # angles = []
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
            # angles.append(observation[4:28])
            result.append(step_func(observation=observation))
            if done:
                results.append(done_func(index=i, result=result))
                break
    # angles = np.array(angles)
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     for j in range(4):
    #         index = i * 4 + j
    #         plt.plot(angles[:, index], label=str(index))
    #     plt.legend()
    # plt.savefig('observation.png')
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
