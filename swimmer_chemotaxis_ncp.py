""" train NCP swimmer of chemotaxis behavior with offline testing """

import multiprocessing
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.networks.ncp.ltc_cell import LTCCell
from src.networks.ncp.rnn_sequence import RNNSequence
from src.networks.ncp.wirings import FullyConnected


def prepare_data(eval_ratio, test_ratio, batch_size, seed):
    """ load data and prepare data loaders """
    dataset = torch.load('data/concat_ncp.pt')
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


def prepare_model(device, units, output_dim, in_features, model_path=None):
    wiring = FullyConnected(units=units, output_dim=output_dim)
    ltc_cell = LTCCell(wiring, in_features=in_features)
    model = RNNSequence(ltc_cell)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model


def train(model, device, loader, criterion, optimizer, writer, epoch):
    model.train()
    mse = 0.
    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        mse += loss.item() * len(data)
        writer.add_scalar('Loss/train', loss.item(), epoch * len(loader) + i)
    mse /= len(loader.dataset)
    return mse


def test(model, device, loader, criterion):
    model.eval()
    mse = 0.
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            mse += loss.item() * len(data)
    mse /= len(loader.dataset)
    return mse


def train_and_eval(model, device, writer, train_loader, eval_loader, optimizer, epochs, early_stop, criterion, save_dir):
    save_path = os.path.join(save_dir, 'model.pt')
    mse_best, e_best = 100., 0
    for e in tqdm(range(epochs)):
        # train
        mse = train(model, device, train_loader, criterion, optimizer, writer, epoch=e)
        writer.add_scalar('MSE/train', mse, e)
        # eval
        mse = test(model, device, eval_loader, criterion)
        writer.add_scalar('MSE/eval', mse, e)
        # eval best
        if mse < mse_best:
            torch.save(model.state_dict(), save_path)
            mse_best, e_best = mse, e
            writer.add_scalar('MSE/best', mse_best, e_best)
        # early stop
        if e - e_best >= early_stop:
            print('early stop')
            break
    return save_path


def main(
    eval_ratio=0.15, test_ratio=0.15, batch_size=512, seed=42, cuda=0, units=19, output_dim=11, in_features=2, lr=0.01,
    epochs=200, early_stop=50
):
    """
    eval_ratio: ratio of eval dataset to the whole dataset
    test_ratio: ratio of test dataset to the whole dataset
    seed: reproducibility on splitting dataset
    units: total amount of neurons (excluding input neurons)
    output_dim: amount of neurons that also act as output
    in_features: input channel amount
    """
    torch.manual_seed(seed)
    writer = SummaryWriter(comment='swimmer_chemotaxis_ncp')
    train_loader, eval_loader, test_loader = prepare_data(eval_ratio, test_ratio, batch_size, seed)
    device = torch.device('cuda:{}'.format(cuda) if torch.cuda.is_available() else 'cpu')
    # train
    model = prepare_model(device, units, output_dim, in_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction='mean')
    save_path = train_and_eval(model, device, writer, train_loader, eval_loader, optimizer, epochs, early_stop, criterion, save_dir=writer.log_dir)
    # test
    model = prepare_model(device, units, output_dim, in_features, model_path=save_path)
    mse = test(model, device, test_loader, criterion)
    # hparams and results
    writer.add_hparams(
        {
            'eval_ratio': eval_ratio, 'test_ratio': test_ratio, 'batch_size': batch_size, 'seed': seed, 'cuda': cuda,
            'units': units, 'output_dim': output_dim, 'in_features': in_features, 'lr': lr
        },
        {'hparam/MSE/test': mse}
    )
    writer.close()


if __name__ == '__main__':
    main(cuda=0, epochs=200)
