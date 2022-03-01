""" test NCP swimmer of chemotaxis behavior with online/active testing """

import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from virtual_nematode.data.simulation import SimulationDataset
from virtual_nematode.data.concat import ConcatDataset
from virtual_nematode.utils import clock_position, sample_seed
from sim import make_swimmer
from virtual_nematode.trainers.ncp import prepare_model


def online_test_single_simulation(env, model, dataset):
    """ run a chemotaxis simulation controlled by a network model """
    seed = sample_seed()
    torch.manual_seed(seed)  # seed model
    observation = env.reset(seed=seed, return_info=False)
    model.eval()
    hidden_state = None
    y = []
    concentrations = []
    with torch.no_grad():
        for i in range(10 ** 6):
            # env.render()
            data = dataset.encode_input_func(g=observation[27])  # encode gradient: []->[2]
            data = data.unsqueeze(dim=0)  # add batch dimension: [2]->[1, 2]
            output, hidden_state = model.step(data, hidden_state)
            action = output.squeeze(dim=0)  # squeeze batch dimension
            action = action.numpy()
            y.append(action.tolist())
            observation, reward, done, info = env.step(action)
            concentrations.append(observation[26])
            if done:
                break
    env.close()
    x = torch.tensor(concentrations, dtype=torch.float64).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float64)
    return x, y


def online_test(
    seed=42, max_episode_steps=2500, distance=15, model_folder=None, model_name='fully_connected', data_name='source.pt',
    data_size=1200
):
    """
    data_size: should be at least 100 trials for each env
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    assert model_folder is not None, 'model_folder can not be {}'.format(model_folder)
    model_dir = os.path.join('runs', model_folder)
    writer = SummaryWriter(log_dir=model_dir)
    envs = [make_swimmer(max_episode_steps=max_episode_steps, x=x, y=y) for x, y in clock_position(distance)]
    model = prepare_model(model_name, model_path=os.path.join(model_dir, 'model.pt'))
    dataset = torch.load(os.path.join('data', data_name))
    data_size = data_size // len(envs)
    results = []
    for env in envs:
        action_size = env.action_space.shape[0]
        result = SimulationDataset(
            data_size, max_episode_steps, 1, action_size, seed, online_test_single_simulation,
            env=env, model=model, dataset=dataset
        )
        results.append(result)
    results = ConcatDataset(results)
    print('results', len(results), results[0][0].size(), results[0][1].size())
    chemotaxis_index = torch.mean(results.tensors[0]).item()
    print('chemotaxis index mean', chemotaxis_index)
    writer.add_hparams(
        {
            'seed': seed, 'max_episode_steps': max_episode_steps, 'distance': distance, 'model_name': model_name,
            'model_dir': model_dir, 'data_size': data_size
        },
        {'hparam/ChemotaxisIndexMean/online': chemotaxis_index}
    )
    writer.close()


def online_test_video(seed=42, max_episode_steps=2500, distance=15, model_folder=None, model_name='fully_connected', data_name='source.pt'):
    """ online test and record video """
    np.random.seed(seed)
    torch.manual_seed(seed)
    assert model_folder is not None, 'model_folder can not be {}'.format(model_folder)
    env = make_swimmer(max_episode_steps=max_episode_steps, x=distance, y=0, camera_name='track', video_name=model_folder)
    model = prepare_model(model_name, model_path=os.path.join('runs', model_folder, 'model.pt'))
    dataset = torch.load(os.path.join('data', data_name))
    x, _ = online_test_single_simulation(env, model, dataset)
    print('chemotaxis index', torch.mean(x).item())


if __name__ == '__main__':
    model_folder = None
    model_name = 'fully_connected'
    data_name = 'ncp.pt'
    online_test(data_size=1200, model_folder=model_folder, model_name=model_name, data_name=data_name)
    online_test_video(model_folder=model_folder, model_name=model_name, data_name=data_name)
