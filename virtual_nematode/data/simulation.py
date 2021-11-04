import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
from virtual_nematode.utils import sample_seed


class SimulationSample(torch.utils.data.Dataset):
    """ run a simulation as a sample """
    def __init__(self, data_size, get_item, **kwargs):
        super(SimulationSample, self).__init__()
        self.data_size = data_size
        self.get_item = get_item  # fn of simulation
        self.kwargs = kwargs  # kwargs for simulation

    def __getitem__(self, index):
        return self.get_item(**self.kwargs)

    def __len__(self):
        return self.data_size


class SimulationDataset(torch.utils.data.TensorDataset):
    """ run simulations to generate dataset """
    def __init__(self, data_size, max_episode_steps, input_size, action_size, seed, get_item, **kwargs):
        self.data_size = data_size
        self.max_episode_steps = max_episode_steps
        self.input_size = input_size  # input features size
        self.action_size = action_size
        """ seeding """
        if seed is not None:
            np.random.seed(seed)
        """ dataset """
        x, y = self.get_tensors(get_item, **kwargs)
        super(SimulationDataset, self).__init__(x, y)

    @staticmethod
    def worker_init_fn(worker_id):
        """ init fn for multiprocessing workers
        seed: seed + worker_id to generate different seeds for different workers
        """
        _, keys, _, _, _ = np.random.get_state()
        seed = keys[0]
        np.random.seed(np.uint32(seed + worker_id))

    def get_tensors(self, get_item, **kwargs):
        """ generate a sample dataset
        x: torch.Tensor, (data_size, max_episode_steps)
        y: torch.Tensor, (data_size, max_episode_steps, action_size)
        """
        data_sample = SimulationSample(self.data_size, get_item, **kwargs)
        dataloader = torch.utils.data.DataLoader(
            data_sample, batch_size=1, shuffle=False, num_workers=multiprocessing.cpu_count(),
            worker_init_fn=self.worker_init_fn
        )
        x = torch.zeros(self.data_size, self.max_episode_steps, self.input_size, dtype=torch.float64)
        y = torch.zeros(self.data_size, self.max_episode_steps, self.action_size, dtype=torch.float64)
        for i, (sample_x, sample_y) in enumerate(tqdm(dataloader)):
            x[i] = sample_x.squeeze(dim=0)
            y[i] = sample_y.squeeze(dim=0)
        return x, y


def generate_sample(env, model, model_kwargs_func, x_func, y_func):
    """ run a forward movement simulation
    x: torch.Tensor, (max_episode_steps, x_size)
    y: torch.Tensor, (max_episode_steps, action_size)
    model_kwargs_func: function, take in observation and return the needed kwargs for model.step()
    x_func: function, return x
    y_func: function, return y
    """
    seed = sample_seed()
    env.seed(seed)  # seed env
    model.seed(seed)  # seed model
    observation = env.reset()
    x = []
    y = []
    for i in range(10 ** 6):
        action = model.step(step=i, **model_kwargs_func(observation=observation))
        stimuli = model.stimuli(step=i)
        x.append(x_func(stimuli=stimuli, observation=observation))
        y.append(y_func(action=action))
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    return x, y


def generate_dataset(env, model, model_kwargs_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed):
    """ generate forward movement dataset
    input_size: x size
    """
    dataset = SimulationDataset(
        data_size, max_episode_steps, input_size, action_size, seed, generate_sample,
        # simulation fn kwargs
        env=env, model=model, model_kwargs_func=model_kwargs_func, x_func=x_func, y_func=y_func
    )
    print('dataset', len(dataset), dataset[0][0].size(), dataset[0][1].size())
    return dataset
