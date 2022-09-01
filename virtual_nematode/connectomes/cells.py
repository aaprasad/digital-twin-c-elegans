import pandas as pd


def head_motor_neurons():
    """ head motor neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    cells = [
        'URADL', 'URADR', 'URAVL', 'URAVR',
        'RMEL', 'RMER', 'RMED', 'RMEV',
        'RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR',
        'RIVL', 'RIVR',
        'RMHL', 'RMHR'
    ]
    return cells


def sublateral_motor_neurons():
    """ sublateral motor neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    cells = [
        'SABD', 'SABVL', 'SABVR',
        'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR',
        'SMBDL', 'SMBDR', 'SMBVL', 'SMBVR',
        'SIBDL', 'SIBDR', 'SIBVL', 'SIBVR',
        'SIADL', 'SIADR', 'SIAVL', 'SIAVR'
    ]
    return cells


def body_wall_muscles():
    """ body wall muscles
    https://doi.org/10.1038/s41586-019-1352-7
    """
    cells = []
    for quadrant in ['dL', 'dR', 'vL', 'vR']:
        for i in range(1, 25):
            if i == 24 and quadrant == 'vL':
                continue
            cells.append(quadrant[0] + 'BWM' + quadrant[1] + str(i))
    return cells


def vb_motor_neurons():
    """ ventral B-type motor neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    return ['VB{:02d}'.format(i) for i in range(1, 12)]


def db_motor_neurons():
    """ dorsal B-type motor neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    return ['DB{:02d}'.format(i) for i in range(1, 8)]


def va_motor_neurons():
    """ ventral A-type motor neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    return ['VA{:02d}'.format(i) for i in range(1, 13)]


def da_motor_neurons():
    """ dorsal A-type motor neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    return ['DA{:02d}'.format(i) for i in range(1, 10)]


def vd_motor_neurons():
    """ ventral D-type motor neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    return ['VD{:02d}'.format(i) for i in range(1, 14)]


def dd_motor_neurons():
    """ dorsal D-type motor neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    return ['DD{:02d}'.format(i) for i in range(1, 7)]


def neuron_list():
    """ head motor neurons + vnc motor neurons + sublateral motor neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    # head motor neurons: URA, RME, RMD, RIV, RMH
    head = head_motor_neurons()
    # ventral cord motor neurons: VA, DA, VB, DB, VD, DD, AS
    vnc = vb_motor_neurons() + db_motor_neurons() + vd_motor_neurons() + dd_motor_neurons()
    # sublateral motor neurons: SAB, SMD, SMB, SIB, SIA
    sublateral = sublateral_motor_neurons()
    # all cells
    cells = head + vnc + sublateral
    return cells


def neuron_list1():
    """ include AVB interneurons
    https://doi.org/10.1098/rstb.2017.0379
    https://doi.org/10.1073/pnas.1717022115
    https://doi.org/10.1038/s41598-021-92690-2
    """
    cells = ['AVBL', 'AVBR'] + neuron_list()
    return cells


def neuron_list2(path, muscles):
    """ all neurons
    https://doi.org/10.1038/s41586-019-1352-7
    gap_junction.index and gap_junction.columns contain the same set of cells
    chemical.index is a subset of gap_junction.index
    chemical.columns (excluding {'g2L', 'bm', 'g2R', 'g1p'}) is a subset of gap_junction.index
    since these 4 cells only receive chemical input and don't generate output, they can be discarded
    """
    gap_junction = pd.read_excel(path, sheet_name='hermaphrodite gap jn symmetric', header=2, index_col=2).iloc[:469, 2:471]
    all_cells = list(gap_junction.index)
    muscles_cells = set(muscles)
    cells = []
    for cell in all_cells:
        if cell not in muscles_cells:
            cells.append(cell)
    return cells


def neuron_list3(path):
    """ all cells excluding body wall muscles, other end organs, sex-specific cells (motor neurons and muscles) """
    gap_junction = pd.read_excel(path, sheet_name='hermaphrodite gap jn symmetric', header=2, index_col=2).iloc[:469, 2:471]
    cells = list(gap_junction.iloc[0:329, :].index)  # 329 cells
    return cells


def sensory_neurons(path):
    """ all sensory neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    gap_junction = pd.read_excel(path, sheet_name='hermaphrodite gap jn symmetric', header=2, index_col=2).iloc[:469, 2:471]
    cells = list(gap_junction.iloc[57:140, :].index)
    return cells


def cell_list(path):
    """ all cells
    374 neurons + 95 body wall muscles -> 469 cells
    """
    muscles = body_wall_muscles()
    neurons = neuron_list2(path, muscles)
    cells = neurons + muscles
    return cells


def motor_neurons(path):
    """ all motor neurons
    https://doi.org/10.1038/s41586-019-1352-7
    """
    gap_junction = pd.read_excel(path, sheet_name='hermaphrodite gap jn symmetric', header=2, index_col=2).iloc[:469, 2:471]
    cells = list(gap_junction.iloc[221:329, :].index)
    return cells


def ventral_cord_motor_neurons(path):
    """ https://doi.org/10.1038/s41586-019-1352-7 """
    gap_junction = pd.read_excel(path, sheet_name='hermaphrodite gap jn symmetric', header=2, index_col=2).iloc[:469, 2:471]
    cells = list(gap_junction.iloc[258:329, :].index)
    return cells
