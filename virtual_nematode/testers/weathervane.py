import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from virtual_nematode.data.simulation import SimulationDataset
from virtual_nematode.testers.forward import test_func2


def tester(env, model, data_func, x_func, y_func, seed, max_episode_steps, model_folder, model_name, data_size=100):
    """ online test for at least 100 trials with torch multiprocessing """
    np.random.seed(seed)
    torch.manual_seed(seed)
    writer = SummaryWriter(log_dir=model_folder)
    x_func_size = 8  # size of x_func's return
    action_size = env.action_space.shape[0]
    result = SimulationDataset(
        data_size, max_episode_steps, x_func_size, action_size, seed, test_func2, disable=False,
        # func kwargs
        env=env, model=model, data_func=data_func, x_func=x_func, y_func=y_func
    )
    x, y = result.tensors
    chemotaxis_index_mean = x[:, :, 4].mean().item()
    start_concentration_mean = x[:, 0, 4].mean().item()
    print(
        'chemotaxis index mean {:.2f}, start concentration mean {:.2f} / {} steps'.format(
            chemotaxis_index_mean, start_concentration_mean, max_episode_steps
        )
    )
    hparam = {
        'seed': seed, 'max_episode_steps': max_episode_steps, 'model_folder': model_folder, 'model_name': model_name,
        'data_size': data_size
    }
    writer.add_hparams(
        hparam,
        {
            'hparam/ChemotaxisIndexMean/online': chemotaxis_index_mean,
            'hparam/StartConcentrationMean/online': start_concentration_mean
        }
    )
    writer.close()
    return x, y  # x_func's return, y_func's return (action)


def single_tester(env, model, data_func, x_func, y_func, seed, max_episode_steps):
    """ online test for a single trial and record video """
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y = test_func2(0, env, model, data_func, x_func, y_func)
    chemotaxis_index = x[:, 4].mean().item()
    start_concentration = x[0, 4].item()
    print('chemotaxis index {:.2f}, start concentration {:.2f} / {} steps'.format(chemotaxis_index, start_concentration, max_episode_steps))
    return x, y  # x_func's return, y_func's return (action)
