import multiprocessing
import numpy as np
import torch
from tqdm import tqdm


class ChemotaxisDataSample(torch.utils.data.Dataset):
    def __init__(self, env, model, data_size):
        super(ChemotaxisDataSample, self).__init__()
        self.env = env
        self.model = model
        self.data_size = data_size

    def __getitem__(self, index):
        return self.generate_sample(env=self.env, model=self.model)

    def __len__(self):
        return self.data_size

    @staticmethod
    def sample_seed():
        """ sample a seed number
        sample a seed number to seed env and model first, and then generate a new sample
        """
        return np.random.randint(np.iinfo(np.uint32).max)

    def generate_sample(self, env, model):
        """ an env simulation data sample
        x: torch.Tensor, (max_episode_steps, )
        y: torch.Tensor, (max_episode_steps, action_size)
        """
        seed = self.sample_seed()
        env.seed(seed)
        model.seed(seed)
        observation = env.reset()
        info = env.get_info(info={})
        x, y = [], []
        for i in range(10 ** 6):
            # env.render()
            x.append(info['concentration'])
            action = model.step(step=i, q=observation[1:12], q_vel=observation[15:], g_p=info['g_p'], g_w=info['g_w'])
            y.append(action.tolist())
            observation, reward, done, info = env.step(action)
            if done:
                break
        env.close()
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


class ChemotaxisDataset(torch.utils.data.TensorDataset):
    """ generate chemotaxis dataset
    x: concentrations sensed at nose tip
    y: actions performed each step
    """
    def __init__(self, env, model, data_size, max_episode_steps, seed):
        self.data_size = data_size
        self.max_episode_steps = max_episode_steps
        self.action_size = env.action_space.shape[0]
        self.source = env.source.tolist()  # help identify the position of chemical source
        """ seeding """
        self.seed(seed)
        """ dataset """
        x, y = self.generate_dataset(env, model)
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

    def generate_dataset(self, env, model):
        """ generate a sample dataset
        seed: seed + worker_id to generate different seeds for different workers
        x: torch.Tensor, (data_size, max_episode_steps)
        y: torch.Tensor, (data_size, max_episode_steps, action_size)
        """
        data_sample = ChemotaxisDataSample(env, model, data_size=self.data_size)
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
