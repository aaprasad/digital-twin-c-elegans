import numpy as np
import torch
from tqdm import tqdm


class ChemotaxisDataset(torch.utils.data.Dataset):
    """ generate chemotaxis dataset """
    def __init__(self, make_env, make_model, data_size, source_pos, seed, env_kwargs):
        self.make_env = make_env  # function
        self.make_model = make_model  # class
        self.env_kwargs = env_kwargs  # env kwargs
        """ seeding """
        self.seed(seed)
        """ dataset """
        self.x, self.y = self.generate_dataset(data_size, source_pos)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)

    @staticmethod
    def seed(seed):
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def sample_seed():
        """ sample a seed number """
        return np.random.randint(np.iinfo(np.uint32).max)

    def generate_sample(self, x_pos, y_pos):
        """ run an env simulation """
        env = self.make_env(x=x_pos, y=y_pos, **self.env_kwargs)
        env.seed(seed=self.sample_seed())
        observation = env.reset()
        info = {'g': 0., 'g_p': 0., 'g_w': 0.}
        model = self.make_model(dt=env.dt, seed=self.sample_seed())
        x, y = [], []
        for i in range(10 ** 6):
            # env.render()
            x.append(info['g'])
            action = model.step(step=i, q=observation[1:12], q_vel=observation[15:], g_p=info['g_p'], g_w=info['g_w'])
            y.append(action)
            observation, reward, done, info = env.step(action)
            if done:
                break
        env.close()
        return x, y

    def generate_dataset(self, data_size, source_pos):
        """ generate a sample dataset
        x: concentrations sensed at nose tip
        y: actions performed each step
        """
        x, y = [], []
        data_size = data_size // len(source_pos) * len(source_pos)
        with tqdm(total=data_size) as pbar:
            for _ in range(data_size // len(source_pos)):
                for x_pos, y_pos in source_pos:
                    g, action = self.generate_sample(x_pos=x_pos, y_pos=y_pos)
                    x.append(g)
                    y.append(action)
                    pbar.update(1)
        return x, y
