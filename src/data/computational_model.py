import multiprocessing
import numpy as np
import torch
from tqdm import tqdm


class ChemotaxisDataSample(torch.utils.data.Dataset):
    def __init__(self, envs, models, data_size, seed):
        super(ChemotaxisDataSample, self).__init__()
        assert len(envs) == len(models), 'The lengths of envs and models must be the same.'
        self.envs = envs
        self.models = models
        self.data_size = data_size
        self.env_amount = len(envs)
        """ seeding """
        self.seed(seed)

    def __getitem__(self, index):
        index = index % self.env_amount
        return self.generate_sample(env=self.envs[index], model=self.models[index])

    def __len__(self):
        return self.data_size

    @staticmethod
    def seed(seed):
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def sample_seed():
        """ sample a seed number """
        return np.random.randint(np.iinfo(np.uint32).max)

    def generate_sample(self, env, model):
        """ an env simulation data sample
        x: torch.Tensor, (max_episode_steps, )
        y: torch.Tensor, (max_episode_steps, action_size)
        """
        seed = self.sample_seed()
        env.seed(seed)
        observation = env.reset()
        info = {'g': 0., 'g_p': 0., 'g_w': 0.}
        model.seed(seed)
        x, y = [], []
        for i in range(10 ** 6):
            # env.render()
            x.append(info['g'])
            action = model.step(step=i, q=observation[1:12], q_vel=observation[15:], g_p=info['g_p'], g_w=info['g_w'])
            y.append(action.tolist())
            observation, reward, done, info = env.step(action)
            if done:
                break
        env.close()
        x = torch.tensor(x, dtype=torch.float64)
        y = torch.tensor(y, dtype=torch.float64)
        return x, y


class ChemotaxisDataset(torch.utils.data.Dataset):
    """ generate chemotaxis dataset
    x: concentrations sensed at nose tip
    y: actions performed each step
    """
    def __init__(self, make_env, make_model, data_size, sources, seed, env_kwargs):
        super(ChemotaxisDataset, self).__init__()
        self.make_env = make_env  # function
        self.make_model = make_model  # class
        self.env_kwargs = env_kwargs  # env kwargs
        """ dataset """
        self.x, self.y = self.generate_dataset(data_size, sources, seed)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def generate_dataset(self, data_size, sources, seed):
        """ generate a sample dataset
        x: torch.Tensor, (data_size, max_episode_steps)
        y: torch.Tensor, (data_size, max_episode_steps, action_size)
        """
        x, y = [], []
        envs = [self.make_env(x=pos_x, y=pos_y, **self.env_kwargs) for pos_x, pos_y in sources]
        models = [self.make_model(dt=env.dt) for env in envs]
        data_sample = ChemotaxisDataSample(envs, models, data_size=data_size, seed=seed)
        dataloader = torch.utils.data.DataLoader(data_sample, batch_size=1, shuffle=False, num_workers=multiprocessing.cpu_count())
        for sample_x, sample_y in tqdm(dataloader):
            x.append(sample_x.squeeze(dim=0))
            y.append(sample_y.squeeze(dim=0))
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y
