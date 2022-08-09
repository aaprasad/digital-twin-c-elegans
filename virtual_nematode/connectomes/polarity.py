import pandas as pd


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
