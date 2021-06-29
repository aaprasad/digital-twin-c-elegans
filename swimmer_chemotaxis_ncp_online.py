""" test NCP swimmer of chemotaxis behavior with online/active testing """

import numpy as np
import torch
from src.utils import clock_position, sample_seed
from swimmer_chemotaxis import make_swimmer
from swimmer_chemotaxis_ncp import prepare_model


def test_chemotaxis_ncp(env, model, dataset, seed):
    """ run a chemotaxis simulation controlled by a network model """
    env.seed(seed)
    env.reset()
    model.eval()
    info = env.get_info(info={})
    hidden_state = None
    with torch.no_grad():
        for i in range(10 ** 6):
            # env.render()
            x = dataset.encode_input_func(g=info['g'])  # encode gradient
            data = x.unsqueeze(dim=0)  # add batch dimension
            output, hidden_state = model.step(data, hidden_state)
            output = output.squeeze(dim=0)  # squeeze batch dimension
            action = dataset.decode_output_func(output)  # decode output
            action = action.numpy()
            observation, reward, done, info = env.step(action)
            if done:
                break
    print('Chemotaxis index {:.4f}'.format(np.mean(env.stats['concentration'])))
    env.close()


def online_test(seed=42, max_episode_steps=2500, distance=15, units=19, output_dim=11, in_features=2, model_path=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    envs = [make_swimmer(max_episode_steps=max_episode_steps, x=x, y=y) for x, y in clock_position(distance)]
    model = prepare_model(units, output_dim, in_features, model_path=model_path)
    dataset = torch.load('data/ncp.pt')
    test_chemotaxis_ncp(envs[0], model, dataset, seed=sample_seed())


if __name__ == '__main__':
    online_test(model_path=None)
