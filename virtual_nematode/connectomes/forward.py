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


def cell_list():
    """ head motor neurons + vnc motor neurons + sublateral motor neurons + body wall muscles
    https://doi.org/10.1038/s41586-019-1352-7
    """
    # head motor neurons: URA, RME, RMD, RIV, RMH
    head = head_motor_neurons()
    # ventral cord motor neurons: VA, DA, VB, DB, VD, DD, AS
    vnc = ['VB{:02d}'.format(i) for i in range(1, 12)] + \
          ['DB{:02d}'.format(i) for i in range(1, 8)] + \
          ['VD{:02d}'.format(i) for i in range(1, 14)] + \
          ['DD{:02d}'.format(i) for i in range(1, 7)]
    # sublateral motor neurons: SAB, SMD, SMB, SIB, SIA
    sublateral = sublateral_motor_neurons()
    # body wall muscles
    muscles = body_wall_muscles()
    # all cells
    cells = head + vnc + sublateral + muscles
    return cells
