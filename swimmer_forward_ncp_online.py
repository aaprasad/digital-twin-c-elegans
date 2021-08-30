""" online/active testing of NCP swimmer's forward locomotion ability """

import numpy as np
import os
from src.data.simulation import SimulationDataset
from src.models.forward import Forward
from src.utils import sample_seed
from swimmer_chemotaxis_ncp import prepare_model
from swimmer_forward import make_swimmer
import torch
from torch.utils.tensorboard import SummaryWriter


def online_test_single_simulation(env, model, math_model, mode):
    """ run a forward locomotion simulation steered by a neural network """
    seed = sample_seed()
    env.seed(seed)
    torch.manual_seed(seed)
    env.reset()
    model.eval()
    math_model.seed(seed)
    hidden_state = None
    y = []
    with torch.no_grad():
        for i in range(10 ** 6):
            data = math_model.stimuli(step=i, mode=mode)  # external stimulus signal as input
            data = torch.tensor(data)
            data = data.unsqueeze(-1)  # add input feature's dimension: []->[1]
            data = data.unsqueeze(dim=0)  # add batch dimension: [1]->[1, 1]
            output, hidden_state = model.step(data, hidden_state)
            action = output.squeeze(dim=0)  # remove batch dimension
            action = action.numpy()
            observation, reward, done, info = env.step(action)
            y.append(action.tolist())
            if done:
                break
    env.close()
    x = torch.tensor(env.stats['com'], dtype=torch.float32)  # center of mass, (max_episode_steps, 2)
    y = torch.tensor(y, dtype=torch.float32)  # (max_episode_steps, action_size)
    return x, y


def online_test(
    seed=42, max_episode_steps=2500, model_folder=None, model_name='fully_connected', data_size=100, mode='sine_wave',
    **kwargs
):
    """ online test for at least 100 trials """
    np.random.seed(seed)
    torch.manual_seed(seed)
    assert model_folder is not None, 'model_folder can not be {}'.format(model_folder)
    model_dir = os.path.join('runs', model_folder)
    writer = SummaryWriter(log_dir=model_dir)
    env = make_swimmer(max_episode_steps=max_episode_steps)
    model = prepare_model(model_name, model_path=os.path.join(model_dir, 'model.pt'), **kwargs)
    math_model = Forward(dt=env.dt, seed=seed)
    action_size = env.action_space.shape[0]
    result = SimulationDataset(
        data_size, max_episode_steps, 2, action_size, seed, online_test_single_simulation,
        # simulation fn kwargs
        env=env, model=model, math_model=math_model, mode=mode
    )
    print('result', len(result), result[0][0].size(), result[0][1].size())
    x, _ = result.tensors
    displacement_mean = torch.linalg.norm(x[:, -1, :] - x[:, 0, :], ord=2, dim=1).mean().item()
    print('com displacement mean {:.2f} / {} steps'.format(displacement_mean, max_episode_steps))
    hparam = {
        'seed': seed, 'max_episode_steps': max_episode_steps, 'model_folder': model_folder, 'model_name': model_name,
        'data_size': data_size, 'mode': mode
    }
    for key in kwargs:
        hparam[key] = kwargs.get(key)
    writer.add_hparams(hparam, {'hparam/DisplacementMean/online': displacement_mean})
    writer.close()


def online_test_video(seed=42, max_episode_steps=2500, model_folder=None, model_name='fully_connected', mode='sine_wave', **kwargs):
    """ online test and record video """
    np.random.seed(seed)
    torch.manual_seed(seed)
    assert model_folder is not None, 'model_folder can not be {}'.format(model_folder)
    env = make_swimmer(max_episode_steps=max_episode_steps, camera_name='track', video_name=model_folder)
    model = prepare_model(model_name, model_path=os.path.join('runs', model_folder, 'model.pt'), **kwargs)
    math_model = Forward(dt=env.dt, seed=seed)
    x, _ = online_test_single_simulation(env, model, math_model, mode)
    displacement = torch.linalg.norm(x[-1, :] - x[0, :], ord=2).item()
    print('com displacement {:.2f} / {} steps'.format(displacement, max_episode_steps))


if __name__ == '__main__':
    model_folder = None
    model_name = 'fully_connected'
    mode = 'sine_wave'
    kwargs = {'units': 30, 'output_dim': 11, 'in_features': 1}
    online_test(data_size=100, model_folder=model_folder, model_name=model_name, mode=mode, **kwargs)
    online_test_video(model_folder=model_folder, model_name=model_name, mode=mode, **kwargs)
