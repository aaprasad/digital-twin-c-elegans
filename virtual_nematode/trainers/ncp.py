import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from virtual_nematode.data.utils import prepare_dataloader, prepare_test_dataloader
from virtual_nematode.networks.ncp.ltc_cell import LTCCell
from virtual_nematode.networks.ncp.rnn_cell_fused import RNNCell2Stage
from virtual_nematode.networks.ncp.rnn_sequence import RNNSequence
from virtual_nematode.networks.ncp.wirings import FullyConnected, NCP
from virtual_nematode.networks.rnn.ctrnn_cell import CTRNNCell
from virtual_nematode.networks.rnn.rnn import RNNCell
from virtual_nematode.networks.snn.forward import SNNCell, SNN, SNNCell1, SNNCell2, SNNCell3, SNNCell4
from virtual_nematode.networks.snn.weathervane import SNNCell as SNNCellW
from virtual_nematode.networks.snn.weathervane import SNNCell1 as SNNCellW1
from virtual_nematode.networks.snn.weathervane import SNNCell3 as SNNCellW3


def prepare_model(model_name, device=None, device_ids=None, model_path=None, strict=True, **kwargs):
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
    elif model_name == 'ctrnn':
        model = ctrnn(**kwargs)
    elif model_name == 'rnn':
        model = rnn(**kwargs)
    elif model_name == 'ctrnn_2_stage':
        model = ctrnn_2_stage(**kwargs)
    elif model_name == 'snn_forward':
        model = snn_forward(**kwargs)
    elif model_name == 'snn_forward1':
        model = snn_forward1(**kwargs)
    elif model_name == 'snn_forward2':
        model = snn_forward2(**kwargs)
    elif model_name == 'snn_forward3':
        model = snn_forward3(**kwargs)
    elif model_name == 'snn_forward4':
        model = snn_forward4(**kwargs)
    elif model_name == 'snn_weathervane':
        model = snn_weathervane(**kwargs)
    elif model_name == 'snn_weathervane1':
        model = snn_weathervane1(**kwargs)
    elif model_name == 'snn_weathervane3':
        model = snn_weathervane3(**kwargs)
    else:
        raise AssertionError('{} not exist'.format(model_name))
    if device is None:
        device = torch.device('cpu')
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=strict)
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


def ctrnn(input_size, hidden_size, output_size, **kwargs):
    cell = CTRNNCell(input_size, hidden_size, output_size, **kwargs)
    model = RNNSequence(cell)
    return model


def ctrnn_2_stage(input_size, hidden_size, output_size, load_kwargs, input1_range, input2_range, load_path=None, freeze=True, **kwargs):
    """ CTRNN 2 stage
    load_kwargs: kwargs for the model to be loaded
    load_path: path to load the model
    freeze: if True, freeze cell 2's params
    """
    # create cells
    cell1 = CTRNNCell(input_size, hidden_size, output_size, **kwargs)
    cell2 = CTRNNCell(**load_kwargs)
    if load_path is not None:
        # load state dict
        state_dict = torch.load(load_path)
        cell2.linear[0].weight.data = state_dict.get('rnn_cell.linear.0.weight')
        cell2.linear[0].bias.data = state_dict.get('rnn_cell.linear.0.bias')
        if freeze is True:
            cell2.requires_grad_(False)
    # create
    cell = RNNCell2Stage(cell1, cell2, input1_range=input1_range, input2_range=input2_range)
    model = RNNSequence(cell)
    return model


def rnn(input_size, hidden_size, output_size, **kwargs):
    cell = RNNCell(input_size, hidden_size, output_size, **kwargs)
    model = RNNSequence(cell)
    return model


def snn_forward(**kwargs):
    cell = SNNCell(**kwargs)
    model = SNN(cell)
    return model


def snn_forward1(**kwargs):
    cell = SNNCell1(**kwargs)
    model = SNN(cell)
    return model


def snn_forward2(**kwargs):
    cell = SNNCell2(**kwargs)
    model = SNN(cell)
    return model


def snn_forward3(**kwargs):
    cell = SNNCell3(**kwargs)
    model = SNN(cell)
    return model


def snn_forward4(**kwargs):
    cell = SNNCell4(**kwargs)
    model = SNN(cell)
    return model


def snn_weathervane(**kwargs):
    cell = SNNCellW(**kwargs)
    model = SNN(cell)
    return model


def snn_weathervane1(**kwargs):
    cell = SNNCellW1(**kwargs)
    model = SNN(cell)
    return model


def snn_weathervane3(**kwargs):
    cell = SNNCellW3(**kwargs)
    model = SNN(cell)
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


def test_none_reduction(model, device, loader, criterion):
    model.eval()
    mse = None
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            if mse is None:
                mse = torch.zero_like(loss).sum(dim=0)
            mse += loss.sum(dim=0)
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
        # module and optimizer state dict
        if type(model) is torch.nn.DataParallel:  # unwrap DataParallel to get module and save state dict
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, '{}{}.pt'.format(model_path[0:-3], e))
        torch.save(optimizer.state_dict(), '{}{}.optim.pt'.format(model_path[0:-3], e))
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
    data_name, model_name, batch_size, seed, device_ids, lr, weight_decay, epochs, early_stop,
    model_path=None, strict=True, optimizer_path=None, **kwargs
):
    """ offline train, eval and test
    data_name: ['train.pt', 'eval.pt', 'test.pt']
    seed: reproducibility on splitting dataset
    units: total amount of neurons (excluding input neurons)
    output_dim: amount of neurons that also act as output
    in_features: input channel amount
    **kwargs: neural network model args
    """
    torch.manual_seed(seed)
    writer = SummaryWriter()
    data_path = [os.path.join('data', name) for name in data_name]
    train_loader, eval_loader, test_loader = prepare_dataloader(data_path, batch_size)
    device = torch.device('cuda:{}'.format(device_ids[0]) if torch.cuda.is_available() else 'cpu')
    model = prepare_model(model_name, device, device_ids, model_path, strict, **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_path is not None:
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    criterion = torch.nn.MSELoss(reduction='mean')
    model_path = os.path.join(writer.log_dir, 'model.pt')
    train_eval(model, device, writer, train_loader, eval_loader, optimizer, epochs, early_stop, criterion, model_path)


def offline_test(data_name, model_name, model_folder, ckpt_name, batch_size, device_ids, save_name, **kwargs):
    data_path = os.path.join('data', data_name)
    test_loader = prepare_test_dataloader(data_path, batch_size)
    device = torch.device('cuda:{}'.format(device_ids) if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(model_folder, ckpt_name)
    model = prepare_model(model_name, device, device_ids, model_path, strict=True, **kwargs)
    # criterion = torch.nn.MSELoss(reduction='mean')
    # mse = test(model, device, test_loader, criterion)
    # print('{:.4e}'.format(mse))
    criterion = torch.nn.MSELoss(reduction='none')
    mse = test_none_reduction(model, device, test_loader, criterion)
    print('{:.4e}'.format(mse.mean().item()))
    torch.save(mse, os.path.join('data', model_folder, ckpt_name, save_name))
