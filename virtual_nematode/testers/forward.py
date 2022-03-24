import numpy as np
from virtual_nematode.data.simulation import SimulationDataset
from virtual_nematode.utils import sample_seed
import torch
from torch.utils.tensorboard import SummaryWriter


def test_func(index, env, model, data_func, x_func):
    """ run a forward locomotion simulation steered by a neural network """
    if type(env) is list:  # env is a list, take the one according to item index
        env = env[index]
    seed = sample_seed()
    torch.manual_seed(seed)
    env.seed(seed)
    observation = env.reset()
    model.eval()
    hidden_state = None
    x, y = [], []
    with torch.no_grad():
        for i in range(10 ** 6):
            # env.render()
            data = data_func(observation=observation)
            data = torch.tensor(data, dtype=torch.float64)
            data = data.unsqueeze(dim=0)  # add batch dimension
            output, hidden_state = model.step(data, hidden_state)
            action = output.squeeze(dim=0)  # remove batch dimension
            action = action.numpy()
            observation, reward, done, info = env.step(action)
            x.append(x_func(observation=observation))
            y.append(action.tolist())
            if done:
                break
    env.close()
    x = torch.tensor(x, dtype=torch.float64)  # center of mass, (max_episode_steps, 2)
    y = torch.tensor(y, dtype=torch.float64)  # (max_episode_steps, action_size)
    return x, y


def tester(
    env, model, data_func, x_func, seed=42, max_episode_steps=2500, model_folder=None, model_name='fully_connected',
    data_size=100, disable=False
):
    """ online test for at least 100 trials with torch multiprocessing """
    np.random.seed(seed)
    torch.manual_seed(seed)
    writer = SummaryWriter(log_dir=model_folder)
    action_size = env.action_space.shape[0]
    result = SimulationDataset(
        data_size, max_episode_steps, 2, action_size, seed, test_func, disable,
        # func kwargs
        env=env, model=model, data_func=data_func, x_func=x_func
    )
    if disable is False:
        print('result', len(result), result[0][0].size(), result[0][1].size())
    x, y = result.tensors
    displacement_mean = torch.linalg.norm(x[:, -1, :] - x[:, 0, :], ord=2, dim=1).mean().item()
    print('com displacement mean {:.2f} / {} steps'.format(displacement_mean, max_episode_steps))
    hparam = {
        'seed': seed, 'max_episode_steps': max_episode_steps, 'model_folder': model_folder, 'model_name': model_name,
        'data_size': data_size
    }
    writer.add_hparams(hparam, {'hparam/DisplacementMean/online': displacement_mean})
    writer.close()
    return x, y  # center of mass, action


def single_tester(env, model, data_func, x_func, seed=42, max_episode_steps=2500):
    """ online test for a single trial and record video """
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y = test_func(env, model, data_func, x_func)
    displacement = torch.linalg.norm(x[-1, :] - x[0, :], ord=2).item()
    print('com displacement {:.2f} / {} steps'.format(displacement, max_episode_steps))
    return x, y  # center of mass, action
