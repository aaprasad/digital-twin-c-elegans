from .sample import Sample
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm


class ForwardDataset(torch.utils.data.TensorDataset):
    """ generate a forward TensorDataset
    x: action sequence of the first joint in trials
    y: action sequences in trials
    """
    def __init__(self, data_size, max_episode_steps, action_size, seed, get_item, **kwargs):
        self.data_size = data_size
        self.max_episode_steps = max_episode_steps
        self.action_size = action_size
        """ seeding """
        if seed is not None:
            np.random.seed(seed)
        x, y = self.get_tensors(get_item, **kwargs)
        super(ForwardDataset, self).__init__(x, y)

    @staticmethod
    def worker_init_fn(worker_id):
        _, keys, _, _, _ = np.random.get_state()
        seed = keys[0]
        np.random.seed(np.uint32(seed + worker_id))

    def get_tensors(self, get_item, **kwargs):
        data_sample = Sample(self.data_size, get_item, **kwargs)
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
