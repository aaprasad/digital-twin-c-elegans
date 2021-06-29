""" test NCP swimmer of chemotaxis behavior with online/active testing """

import numpy as np
import torch
from src.utils import clock_position, sample_seed
from swimmer_chemotaxis import make_swimmer
from swimmer_chemotaxis_ncp import prepare_model


def online_test_single_simulation(env, model, dataset):
    """ run a chemotaxis simulation controlled by a network model """
    seed = sample_seed()
    env.seed(seed)
    torch.manual_seed(seed)
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


def online_test(seed=42, max_episode_steps=2500, distance=15, units=19, output_dim=11, in_features=2, model_path=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    envs = [make_swimmer(max_episode_steps=max_episode_steps, x=x, y=y) for x, y in clock_position(distance)]
    model = prepare_model(units, output_dim, in_features, model_path=model_path)
    dataset = torch.load('data/ncp.pt')
    x, _ = online_test_single_simulation(envs[0], model, dataset)
    print('chemotaxis index', torch.mean(x).item())


if __name__ == '__main__':
    online_test(model_path=None)
