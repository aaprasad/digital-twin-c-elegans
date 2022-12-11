import numpy as np
from virtual_nematode.data.simulation import SimulationDataset
from virtual_nematode.utils import sample_seed
import torch


def test_func(env, model, data_func, x_func, y_func):
    seed = sample_seed()
    torch.manual_seed(seed)
    env.seed(seed)
    observation = env.reset()
    model.eval()
    state, activation = None, None
    x, y = [], []
    with torch.no_grad():
        for i in range(10 ** 6):
            # env.render()
            data = data_func(observation=observation)
            data = torch.tensor(data, dtype=torch.float32)
            data = data.unsqueeze(dim=0)  # add batch dimension
            state, activation, action = model.step(state, activation, data)
            action = action.squeeze(dim=0)  # remove batch dimension
            action = action.numpy()
            x.append(x_func(observation=observation))
            y.append(y_func(state=state, activation=activation, action=action))
            observation, reward, done, info = env.step(action)
            if done:
                break
    env.close()
    x = torch.tensor(x, dtype=torch.float32)  # environment observation, (max_episode_steps, env_obs_size)
    y = torch.tensor(y, dtype=torch.float32)  # network observation, (max_episode_steps, net_obs_size)
    return x, y


def tester(env, model, data_func, x_func, y_func, x_func_size, y_func_size, seed=42, max_episode_steps=2500, data_size=100, num_workders=None):
    """ online test for at least 100 trials with torch multiprocessing """
    np.random.seed(seed)
    torch.manual_seed(seed)
    result = SimulationDataset(
        data_size, max_episode_steps, x_func_size, y_func_size, seed, test_func, num_workders, disable=False,
        # func kwargs
        env=env, model=model, data_func=data_func, x_func=x_func, y_func=y_func
    )
    print('result', len(result), result[0][0].size(), result[0][1].size())
    x, y = result.tensors
    return x, y  # x_func, y_func content


def single_tester(env, model, data_func, x_func, y_func, seed=42):
    """ online test for a single trial """
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y = test_func(env, model, data_func, x_func, y_func)
    return x, y  # x_func, y_func content
