import multiprocessing
import torch


def prepare_dataloader(data_path, lengths, batch_size, seed):
    """ load data and prepare data loaders
    lengths: [train_size, eval_size, test_size]
    """
    dataset = torch.load(data_path)
    train_data, eval_data, test_data = torch.utils.data.random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(seed)
    )
    kwargs = {'drop_last': False, 'num_workers': multiprocessing.cpu_count(), 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
    print('dataset', len(dataset), [len(train_loader.dataset), len(eval_loader.dataset), len(test_loader.dataset)])
    return train_loader, eval_loader, test_loader
