import multiprocessing
import numpy as np
import torch
from tqdm import tqdm


class ChemotaxisDataSample(torch.utils.data.Dataset):
    """ generate a single chemotaxis sample with an env and a model
    x: concentrations sensed at nose tip
    y: actions performed each step
    """
    def __init__(self, env, model, generate_sample, data_size, **kwargs):
        super(ChemotaxisDataSample, self).__init__()
        self.env = env
        self.model = model
        self.generate_sample = generate_sample  # function of chemotaxis simulation
        self.data_size = data_size
        self.kwargs = kwargs

    def __getitem__(self, index):
        return self.generate_sample(env=self.env, model=self.model, **self.kwargs)

    def __len__(self):
        return self.data_size


class ChemotaxisDataset(torch.utils.data.TensorDataset):
    """ generate a chemotaxis TensorDataset with an env and a model
    x: concentrations sensed at nose tip
    y: actions performed each step
    """
    def __init__(self, env, model, generate_sample, data_size, max_episode_steps, seed, **kwargs):
        self.data_size = data_size
        self.max_episode_steps = max_episode_steps
        self.action_size = env.action_space.shape[0]
        self.source = env.source.tolist()  # help identify the position of chemical source
        """ seeding """
        self.seed(seed)
        """ dataset """
        x, y = self.generate_dataset(env, model, generate_sample, **kwargs)
        super(ChemotaxisDataset, self).__init__(x, y)

    @staticmethod
    def seed(seed):
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def worker_init_fn(worker_id):
        _, keys, _, _, _ = np.random.get_state()
        seed = keys[0]
        np.random.seed(np.uint32(seed + worker_id))

    def generate_dataset(self, env, model, generate_sample, **kwargs):
        """ generate a sample dataset
        seed: seed + worker_id to generate different seeds for different workers
        x: torch.Tensor, (data_size, max_episode_steps)
        y: torch.Tensor, (data_size, max_episode_steps, action_size)
        """
        data_sample = ChemotaxisDataSample(env, model, generate_sample, data_size=self.data_size, **kwargs)
        dataloader = torch.utils.data.DataLoader(
            data_sample, batch_size=1, shuffle=False, num_workers=multiprocessing.cpu_count(),
            worker_init_fn=self.worker_init_fn
        )
        x = torch.zeros(self.data_size, self.max_episode_steps, dtype=torch.float32)
        y = torch.zeros(self.data_size, self.max_episode_steps, self.action_size, dtype=torch.float32)
        for i, (sample_x, sample_y) in enumerate(tqdm(dataloader)):
            x[i] = sample_x.squeeze(dim=0)
            y[i] = sample_y.squeeze(dim=0)
        return x, y
