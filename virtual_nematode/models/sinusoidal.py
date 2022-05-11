""" sinusoidal control signal """

import numpy as np


class Sinusoidal(object):
    """ generate sinusoidal control signal
    y = A*sin(omega*t + phi)
    omega = 2*pi*freq
    phi = -2*pi*psi*body_index - pi
    """
    def __init__(self, dt, n, a, freq, psi):
        self.dt = dt  # real time per step
        self.n = n  # number of bodies
        self.a = a  # max control value
        self.omega = 2 * np.pi * freq
        self.psi = psi  # body wavelength (rad)
        self.phi = -2 * np.pi * self.psi * np.arange(0, self.n - 1) - np.pi

    def step(self, step):
        return self.a * np.sin(self.omega * step * self.dt + self.phi)

    def stimuli(self, step):
        """ extra signal used as dataset input """
        return None
