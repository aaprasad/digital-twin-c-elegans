def simulate(env, model, model_kwargs_func, step_func, done_func, seed=None, trials=1, render=False):
    """ forward sinusoidal movement
    model_kwargs_func: function, take in observation and return the needed kwargs for model.step()
    step_func: called after every step, process observation and collect result of 1 simulation
    done_func: called after every simulation, process simulation result and collect results of all simulations
    **kwargs: configure mathematical model
    """
    results = []  # results of all simulations
    if trials > 1:
        seed = None  # ensure that seed is different in each trial
    for i in range(trials):
        result = []  # results of 1 simulation
        env.seed(seed)
        observation = env.reset()
        for step in range(10 ** 6):
            if render is True:
                env.render()
            action = model.step(step=step, **model_kwargs_func(observation=observation))
            observation, reward, done, info = env.step(action)
            result.append(step_func(observation=observation))
            if done:
                results.append(done_func(index=i, result=result))
                break
    return results
