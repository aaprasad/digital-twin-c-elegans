""" A computational model of internal representations of chemical gradients for chemotaxis
- https://doi.org/10.1038/s41598-018-35157-1
"""

import numpy as np


class SinusoidalMotion(object):
    """ oscillator for generating sinusoidal movement """
    def __init__(self):
        self.dt = 0.01
        self.q_max = 40 * np.pi / 180  # max joint angle (rad)
        self.psi = 0.06  # body wavelength (rad)
        self.n = 12  # number of bodies
        self.omega = 2 * np.pi * 1.8  # angular velocity of bending (rad/s): 2 * pi * freq
        self.a_max = 2.  # action: [-a_max, a_max]

    def _joint_angle(self, step):
        """ calculate joint angles """
        phi = -2 * np.pi * self.psi * np.arange(0, self.n - 1)
        q = self.q_max * np.sin(self.omega * step * self.dt - phi)
        return q

    def _action(self, q, q_next, q_vel):
        """ delta_q: [-4 * q_max, 4 * q_max] -> action: [-a_max, a_max]
        - consider (q_vel_next - q_vel) for action
        """
        delta_q = q_next - q - q_vel * self.dt
        action = delta_q / (4 * self.q_max) * self.a_max
        return action

    def step(self, step, q, q_vel):
        q_next = self._joint_angle(step=step)
        action = self._action(q=q, q_next=q_next, q_vel=q_vel)
        return action
