import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from virtual_nematode.data.utils import prepare_dataloader
from virtual_nematode.networks.ncp.ltc_cell import LTCCell
from virtual_nematode.networks.ncp.rnn_sequence import RNNSequence
from virtual_nematode.networks.ncp.wirings import FullyConnected, NCP
from virtual_nematode.trainers.loss import MSESymmetricJointLoss, MSESymmetricMuscleLoss


def prepare_model(model_name, device=None, device_ids=None, model_path=None, **kwargs):
    """ init model
    seed torch rng before initializing model
    device: main torch device
    device_ids: list of GPU ids, should include main device id
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
    if type(device_ids) is list and len(device_ids) >= 2:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    return model


def fully_connected(units=60, output_dim=11, in_features=2, output_mapping='affine'):
    wiring = FullyConnected(units=units, output_dim=output_dim)
    ltc_cell = LTCCell(wiring, in_features=in_features, output_mapping=output_mapping)
    model = RNNSequence(ltc_cell)
    return model


def ncp(
    in_features=2, inter_neurons=24, command_neurons=48, motor_neurons=11, sensory_fanout=12, inter_fanout=5,
    recurrent_command_synapses=6, motor_fanin=4, output_mapping='affine'
):
    wiring = NCP(
        inter_neurons=inter_neurons,
        command_neurons=command_neurons,
        motor_neurons=motor_neurons,
        sensory_fanout=sensory_fanout,
        inter_fanout=inter_fanout,
        recurrent_command_synapses=recurrent_command_synapses,
        motor_fanin=motor_fanin,
    )
    ltc_cell = LTCCell(wiring, in_features=in_features, output_mapping=output_mapping)
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


def train_eval(model, device, writer, train_loader, eval_loader, optimizer, epochs, early_stop, criterion, model_path):
    """ offline train and eval """
    mse_best, e_best = 100., 0
    for e in tqdm(range(epochs)):
        # train
        mse = train(model, device, train_loader, criterion, optimizer, writer, epoch=e)
        writer.add_scalar('MSE/train', mse, e)
        # eval
        mse = test(model, device, eval_loader, criterion)
        writer.add_scalar('MSE/eval', mse, e)
        # state dict
        if type(model) is torch.nn.DataParallel:  # unwrap DataParallel to get module and save state dict
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        # save
        torch.save(state_dict, '{}{}.pt'.format(model_path[0:-3], e))
        # eval best
        if mse < mse_best:
            torch.save(state_dict, model_path)
            mse_best, e_best = mse, e
            writer.add_scalar('MSE/best', mse_best, e_best)
        # early stop
        if e - e_best >= early_stop:
            print('early stop')
            break


def train_eval_test(
    data_name='ncp.pt', model_name='fully_connected', lengths=None, batch_size=2048, seed=42, cuda=0, device_ids=None,
    lr=0.001, weight_decay=0, epochs=200, early_stop=30, comment='', loss='MSELoss', sr=1, **kwargs
):
    """ offline train, eval and test
    lengths: [train_size, eval_size, test_size]
    seed: reproducibility on splitting dataset
    units: total amount of neurons (excluding input neurons)
    output_dim: amount of neurons that also act as output
    in_features: input channel amount
    **kwargs: neural network model args
    """
    torch.manual_seed(seed)
    writer = SummaryWriter(comment=comment)
    data_path = os.path.join('data', data_name)
    train_loader, eval_loader, test_loader = prepare_dataloader(data_path, lengths, batch_size, seed)
    device = torch.device('cuda:{}'.format(cuda) if torch.cuda.is_available() else 'cpu')
    # train
    model = prepare_model(model_name, device, device_ids, **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss == 'MSELoss':
        criterion = torch.nn.MSELoss(reduction='mean')
    elif loss == 'MSESymmetricJointLoss':
        criterion = MSESymmetricJointLoss(reduction='mean', sr=sr)
    elif loss == 'MSESymmetricMuscleLoss':
        criterion = MSESymmetricMuscleLoss(reduction='mean', sr=sr)
    else:
        assert ValueError('Invalid loss type {}'.format(loss))
    model_path = os.path.join(writer.log_dir, 'model.pt')
    train_eval(model, device, writer, train_loader, eval_loader, optimizer, epochs, early_stop, criterion, model_path)
    # test
    model = prepare_model(model_name, device, device_ids, model_path, **kwargs)
    mse = test(model, device, test_loader, criterion)
    # hparams and results
    writer.add_hparams(
        {'lengths': torch.tensor(lengths), 'batch_size': batch_size, 'seed': seed, 'cuda': cuda, 'model_name': model_name, 'lr': lr},
        {'hparam/MSE/test': mse}
    )
    writer.close()
