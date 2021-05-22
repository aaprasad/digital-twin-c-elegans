""" A computational model of internal representations of chemical gradients for chemotaxis
- https://doi.org/10.1038/s41598-018-35157-1
"""

import numpy as np


class SinusoidalMotion(object):
    """ oscillator for generating sinusoidal movement """
    def __init__(self):
        self.dt = 0.01
        self.q_max = 40 * np.pi / 180  # max joint angle (rad)
        self.psi = 1.54  # body wavelength (rad)
        self.n = 12  # number of bodies
        self.omega = 2 * np.pi * 0.8  # angular velocity of bending (rad/s): 2 * pi * freq

    def _joint_angle(self, step):
        """ calculate joint angles """
        phi = -2 * np.pi * self.psi * np.arange(0, self.n - 1)
        q = self.q_max * np.sin(self.omega * step * self.dt - phi)
        return q

    def _action(self, delta_q):
        """ delta_q: [-2 * q_max, 2 * q_max] -> action: [-1, 1] """
        return (delta_q + 2 * self.q_max) / (4 * self.q_max) * 2 - 1

    def step(self, step, q):
        q_next = self._joint_angle(step)
        action = self._action(delta_q=q_next - q)
        return action
