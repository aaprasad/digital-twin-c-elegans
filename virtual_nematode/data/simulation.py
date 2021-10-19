import multiprocessing
import numpy as np
import torch
from tqdm import tqdm


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
