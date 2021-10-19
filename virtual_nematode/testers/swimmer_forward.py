import numpy as np
import os
from virtual_nematode.data.simulation import SimulationDataset
from virtual_nematode.models.forward import Forward
from virtual_nematode.trainers.ncp import prepare_model
from sim import make_swimmer
import torch
from torch.utils.tensorboard import SummaryWriter


def tester(
    test_func, seed=42, max_episode_steps=2500, model_folder=None, model_name='fully_connected', data_size=100, mode='sine_wave',
    **kwargs
):
    """ online test for at least 100 trials (if there's no randomness, 1 trial is enough) """
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
        data_size, max_episode_steps, 2, action_size, seed, test_func,
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
