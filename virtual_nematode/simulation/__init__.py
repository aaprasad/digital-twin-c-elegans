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
    for i in range(trials):
        result = []  # results of 1 simulation
        observation = env.reset(seed=seed, return_info=False)
        for step in range(10 ** 6):
            if render is True:
                env.render()
            action = action_func(model=model, step=step, observation=observation)
            observation, reward, done, info = env.step(action)
            result.append(step_func(observation=observation))
            if done:
                results.append(done_func(index=i, result=result))
                break
    return results
