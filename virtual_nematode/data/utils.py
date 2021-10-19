import multiprocessing
import torch


def prepare_dataloader(data_path, eval_ratio, test_ratio, batch_size, seed):
    """ load data and prepare data loaders """
    dataset = torch.load(data_path)
    eval_size = int(len(dataset) * eval_ratio)
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - eval_size - test_size
    train_data, eval_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, eval_size, test_size], generator=torch.Generator().manual_seed(seed)
    )
    kwargs = {'drop_last': False, 'num_workers': multiprocessing.cpu_count(), 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
    print('dataset', len(dataset), [len(train_loader.dataset), len(eval_loader.dataset), len(test_loader.dataset)])
    return train_loader, eval_loader, test_loader
