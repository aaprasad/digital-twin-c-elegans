import numpy as np
from tqdm import tqdm
from virtual_nematode.utils import sample_seed


def simulate(env, model, action_func, x_func, seed=None, trials=1, render=False):
    """ forward sinusoidal movement
    action_func: called before every step, generate action
    step_func: called after every step, process observation and collect result of 1 simulation
    done_func: called after every simulation, process simulation result and collect results of all simulations
    **kwargs: configure mathematical model
    """
    x, y = [], []
    np.random.seed(seed)  # if seed is None, ignore
    for _ in tqdm(range(trials)):
        observations, actions = [], []
        # env.seed(seed=sample_seed() if seed is not None else None)
        observation, info = env.reset(seed=sample_seed() if seed is not None else None)
        model.reset()
        for step in range(10 ** 6):
            if render is True:
                env.render()
            action = action_func(model=model, step=step, observation=observation)
            observation, reward, terminated, truncated, info = env.step(action)
            observations.append(x_func(observation=observation))
            actions.append(action)
            if terminated or truncated:
                observations, actions = np.array(observations, dtype=np.float32), np.array(actions, dtype=np.float32)
                x.append(observations)
                y.append(actions)
                break
    x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    return x, y


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
