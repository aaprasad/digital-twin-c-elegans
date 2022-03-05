import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from virtual_nematode.data.concat import ConcatDataset
from virtual_nematode.data.simulation import SimulationDataset
from virtual_nematode.testers.forward import test_func


def tester(
    envs, model, data_func, x_func, seed=42, max_episode_steps=2500, model_folder=None, model_name='fully_connected',
    data_size_per_trial=1, disable=False
):
    """ online test for at least 100 times
    x: (trials * data_size_per_trial, max_episode_steps, 1), concentration
    y: (trials * data_size_per_trial, max_episode_steps, action_size), action
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    writer = SummaryWriter(log_dir=model_folder)
    action_size = envs[0].action_space.shape[0]
    result = []
    for env in envs:
        r = SimulationDataset(
            data_size_per_trial, max_episode_steps, 1, action_size, seed, test_func, disable,
            # func kwargs
            env=env, model=model, data_func=data_func, x_func=x_func
        )
        result.append(r)
    result = ConcatDataset(result)
    if disable is False:
        print('result', len(result), result[0][0].size(), result[0][1].size())
    x, y = result.tensors
    chemotaxis_index_mean = x.mean().item()
    start_concentration_mean = x[:, 0].mean().item()
    print('chemotaxis index mean {:.2f}, start concentration mean {:.2f}'.format(chemotaxis_index_mean, start_concentration_mean))
    hparam = {
        'seed': seed, 'max_episode_steps': max_episode_steps, 'model_folder': model_folder, 'model_name': model_name,
        'data_size_per_trial': data_size_per_trial
    }
    writer.add_hparams(
        hparam,
        {
            'hparam/ChemotaxisIndexMean/online': chemotaxis_index_mean,
            'hparam/StartConcentrationMean/online': start_concentration_mean
        }
    )
    writer.close()
    return x, y  # concentration, action


def tester_vector(
    env, model, data_func, x_func, seed=42, max_episode_steps=2500, model_folder=None, model_name='fully_connected'
):
    """ online test for at least 100 times """
    np.random.seed(seed)
    torch.manual_seed(seed)
    writer = SummaryWriter(log_dir=model_folder)
    observation = env.reset(seed=seed, return_info=False)
    model.eval()
    hidden_state = None
    x, y = [], []
    with torch.no_grad():
        for i in range(10 ** 6):
            # env.render()
            data = data_func(observation=observation, vectorized=True)
            data = torch.tensor(data, dtype=torch.float64)
            output, hidden_state = model.step(data, hidden_state)
            action = action.numpy()
            observation, reward, done, info = env.step(action)
            x.append(torch.from_numpy(x_func(observation=observation, vectorized=True)))
            y.append(torch.from_numpy(action))
            if done.all():
                break
    env.close()
    x = torch.stack(x, dim=1, dtype=torch.float64)  # (batch_size, max_episode_steps, 1)
    y = torch.stack(y, dim=1, dtype=torch.float64)  # (batch_size, max_episode_steps, action_size)
    print('result', x.shape, y.shape)
    chemotaxis_index_mean = x.mean().item()
    start_concentration_mean = x[:, 0].mean().item()
    print('chemotaxis index mean {:.2f}, start concentration mean {:.2f}'.format(chemotaxis_index_mean, start_concentration_mean))
    hparam = {
        'seed': seed, 'max_episode_steps': max_episode_steps, 'model_folder': model_folder, 'model_name': model_name,
    }
    writer.add_hparams(
        hparam,
        {
            'hparam/ChemotaxisIndexMean/online': chemotaxis_index_mean,
            'hparam/StartConcentrationMean/online': start_concentration_mean
        }
    )
    writer.close()
    return x, y  # concentration, action


def single_tester(env, model, data_func, x_func, seed=42):
    """ online test for a single trial and record video
    x: (max_episode_steps, 1), concentration
    y: (max_episode_steps, action_size), action
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y = test_func(env, model, data_func, x_func)
    chemotaxis_index = x.mean().item()
    start_concentration = x[0, 0].item()
    print('chemotaxis index {:.2f}, start concentration {:.2f}'.format(chemotaxis_index, start_concentration))
    return x, y
