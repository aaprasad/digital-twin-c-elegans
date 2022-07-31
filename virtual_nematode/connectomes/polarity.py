from virtual_nematode.connectomes.cells import vb_motor_neurons, db_motor_neurons, vd_motor_neurons, dd_motor_neurons, body_wall_muscles


def chemical_synapse_polarity():
    """ excitatory chemical synapses
    https://doi.org/10.1098/rstb.2017.0379
    """
    vb = vb_motor_neurons()
    db = db_motor_neurons()
    vd = vd_motor_neurons()
    dd = dd_motor_neurons()
    muscles = body_wall_muscles()
    excitatory_synapses = [(db, muscles), (vb, muscles), (db, vd), (vb, dd)]
    inhibitory_synapses = [(dd, muscles), (vd, muscles), (db, dd), (vb, vd)]
    return excitatory_synapses, inhibitory_synapses
