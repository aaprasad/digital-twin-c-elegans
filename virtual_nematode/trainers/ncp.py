import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from virtual_nematode.data.utils import prepare_dataloader
from virtual_nematode.networks.ncp.ltc_cell import LTCCell
from virtual_nematode.networks.ncp.rnn_sequence import RNNSequence
from virtual_nematode.networks.ncp.wirings import FullyConnected, NCP


def prepare_model(model_name, device=None, model_path=None, **kwargs):
    """ init model
    seed torch rng before initializing model
    **kwargs: kwargs for initializing model
    """
    if model_name == 'fully_connected':
        model = fully_connected(**kwargs)
    elif model_name == 'ncp':
        model = ncp(**kwargs)
    else:
        raise AssertionError('{} not exist'.format(model_name))
    if device is None:
        device = torch.device('cpu')
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model


def fully_connected(units=60, output_dim=11, in_features=2):
    """ network model
    batch_size: 2048, takes up 9621MiB
    """
    wiring = FullyConnected(units=units, output_dim=output_dim)
    ltc_cell = LTCCell(wiring, in_features=in_features)
    model = RNNSequence(ltc_cell)
    return model


def ncp(
    in_features=2, inter_neurons=24, command_neurons=48, motor_neurons=11, sensory_fanout=12, inter_fanout=5,
    recurrent_command_synapses=6, motor_fanin=4
):
    """ network model
    batch_size: 1024, takes up 8929MiB
    """
    wiring = NCP(
        inter_neurons=inter_neurons,
        command_neurons=command_neurons,
        motor_neurons=motor_neurons,
        sensory_fanout=sensory_fanout,
        inter_fanout=inter_fanout,
        recurrent_command_synapses=recurrent_command_synapses,
        motor_fanin=motor_fanin,
    )
    ltc_cell = LTCCell(wiring, in_features=in_features)
    model = RNNSequence(ltc_cell)
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


def train_and_eval(model, device, writer, train_loader, eval_loader, optimizer, epochs, early_stop, criterion, model_path):
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
            torch.save(model.state_dict(), model_path)
            mse_best, e_best = mse, e
            writer.add_scalar('MSE/best', mse_best, e_best)
        # early stop
        if e - e_best >= early_stop:
            print('early stop')
            break


def offline_train_and_test(
    data_name='ncp.pt', model_name='fully_connected', eval_ratio=0.15, test_ratio=0.15, batch_size=2048, seed=42,
    cuda=0, lr=0.001, epochs=200, early_stop=30, comment='', **kwargs
):
    """
    eval_ratio: ratio of eval dataset to the whole dataset
    test_ratio: ratio of test dataset to the whole dataset
    seed: reproducibility on splitting dataset
    units: total amount of neurons (excluding input neurons)
    output_dim: amount of neurons that also act as output
    in_features: input channel amount
    **kwargs: neural network model args
    """
    torch.manual_seed(seed)
    writer = SummaryWriter(comment=comment)
    data_path = os.path.join('data', data_name)
    train_loader, eval_loader, test_loader = prepare_dataloader(data_path, eval_ratio, test_ratio, batch_size, seed)
    device = torch.device('cuda:{}'.format(cuda) if torch.cuda.is_available() else 'cpu')
    # train
    model = prepare_model(model_name, device, **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction='mean')
    model_path = os.path.join(writer.log_dir, 'model.pt')
    train_and_eval(model, device, writer, train_loader, eval_loader, optimizer, epochs, early_stop, criterion, model_path)
    # test
    model = prepare_model(model_name, device, model_path, **kwargs)
    mse = test(model, device, test_loader, criterion)
    # hparams and results
    writer.add_hparams(
        {
            'eval_ratio': eval_ratio, 'test_ratio': test_ratio, 'batch_size': batch_size, 'seed': seed, 'cuda': cuda,
            'model_name': model_name, 'lr': lr
        },
        {'hparam/MSE/test': mse}
    )
    writer.close()
