""" test NCP swimmer of chemotaxis behavior with online/active testing """

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src.data.chemotaxis import ChemotaxisDataset
from src.data.concat import ConcatDataset
from src.utils import clock_position, sample_seed
from swimmer_chemotaxis import make_swimmer
from swimmer_chemotaxis_ncp import prepare_model


def online_test_single_simulation(env, model, dataset):
    """ run a chemotaxis simulation controlled by a network model """
    seed = sample_seed()
    env.seed(seed)  # seed env
    torch.manual_seed(seed)  # seed model
    env.reset()
    model.eval()
    info = env.get_info(info={})
    hidden_state = None
    y = []
    with torch.no_grad():
        for i in range(10 ** 6):
            # env.render()
            data = dataset.encode_input_func(g=info['g'])  # encode gradient
            data = data.unsqueeze(dim=0)  # add batch dimension
            output, hidden_state = model.step(data, hidden_state)
            output = output.squeeze(dim=0)  # squeeze batch dimension
            action = dataset.decode_output_func(output)  # decode output
            action = action.numpy()
            y.append(action.tolist())
            observation, reward, done, info = env.step(action)
            if done:
                break
    env.close()
    x = torch.tensor(env.stats['concentration'], dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def online_test(
    seed=42, max_episode_steps=2500, distance=15, units=19, output_dim=11, in_features=2, model_path=None,
    data_size=1200
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    writer = SummaryWriter(comment='swimmer_chemotaxis_ncp_online')
    envs = [make_swimmer(max_episode_steps=max_episode_steps, x=x, y=y) for x, y in clock_position(distance)]
    model = prepare_model(units, output_dim, in_features, model_path=model_path)
    dataset = torch.load('data/ncp.pt')
    data_size = data_size // len(envs)
    results = []
    for env in envs:
        action_size = env.action_space.shape[0]
        source = env.source.tolist()
        result = ChemotaxisDataset(
            data_size, max_episode_steps, action_size, source, seed, online_test_single_simulation,
            env=env, model=model, dataset=dataset
        )
        results.append(result)
    results = ConcatDataset(results)
    print('results', len(results), results[0][0].size(), results[0][1].size())
    print(model_path)
    chemotaxis_index = torch.mean(results.tensors[0]).item()
    print('chemotaxis index mean', chemotaxis_index)
    writer.add_hparams(
        {
            'seed': seed, 'max_episode_steps': max_episode_steps, 'distance': distance, 'units': units,
            'output_dim': output_dim, 'in_features': in_features, 'model_path': model_path, 'data_size': data_size
        },
        {'hparam/ChemotaxisIndexMean/online': chemotaxis_index}
    )
    writer.close()


if __name__ == '__main__':
    online_test(data_size=1200, model_path=None)
