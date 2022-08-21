import pandas as pd
from virtual_nematode.connectomes.cells import (
    db_motor_neurons, vb_motor_neurons,
    da_motor_neurons, va_motor_neurons,
    dd_motor_neurons, vd_motor_neurons,
    proprioception_neurons,
    head_motor_neurons, sublateral_motor_neurons,
)


def _chemical_polarities(df):
    source = list(df.loc[:, 'Source'])
    target = list(df.loc[:, 'Target'])
    synapses = [(pre, post) for pre, post in zip(source, target)]
    return synapses


def chemical_polarities(path):
    """ https://doi.org/10.1371/journal.pcbi.1007974 """
    df = pd.read_excel(path)
    ex_synapses = _chemical_polarities(df[df['Sign'] == '+'])
    in_synapses = _chemical_polarities(df[df['Sign'] == '-'])
    return ex_synapses, in_synapses


def _proprioception_polarities(dim, cells):
    """ fully connected """
    synapses = []
    for pre in range(dim):
        for post in cells:
            synapses.append((pre, post))
    return synapses


def proprioception_polarities(dim):
    """ https://doi.org/10.1016/j.neuron.2012.08.039
    The activity of the ventral motor neuron (VB9) is positively correlated with bending toward the ventral side.
    The activity of the dorsal cholinergic neuron (DB6) is positively correlated with bending toward the dorsal side.
    Bending the worm toward the dorsal side activated the DB motor neuron over the VB motor neuron.
    Bending the worm toward the ventral side activated the VB motor neuron over the DB motor neuron.
    nematode lies on its left, and thus dorsal muscles are on its left; ventral muscles are on its right
    if dorsal muscle contracts, joint angle < 0; if ventral muscle contracts, joint angle > 0
    """
    db = db_motor_neurons()
    vb = vb_motor_neurons()
    # da = da_motor_neurons()
    # va = va_motor_neurons()
    # dd = dd_motor_neurons()
    # vd = vd_motor_neurons()
    ex_synapses = _proprioception_polarities(dim, vb)
    in_synapses = _proprioception_polarities(dim, db)
    return ex_synapses, in_synapses


def _full_connections(pre_list, post_list):
    synapses = []
    for pre in pre_list:
        for post in post_list:
            synapses.append((pre, post))
    return synapses


def proprioception_connections(dim):
    synapses = _full_connections(pre_list=list(range(dim)), post_list=proprioception_neurons())
    return synapses, [], []
