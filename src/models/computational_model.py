""" A computational model of internal representations of chemical gradients for chemotaxis
- https://doi.org/10.1038/s41598-018-35157-1
"""

import numpy as np


class SinusoidalMotion(object):
    """ oscillator for generating sinusoidal movement """
    def __init__(self):
        self.dt = 0.01
        self.n = 12  # number of bodies
        self.q_max = 40 * np.pi / 180  # max joint angle (rad)
        """ affect sinusoidal posture and speed """
        self.a_max = 2.  # action: [-a_max, a_max]
        self.psi = 0.2  # body wavelength (rad)
        self.omega = 2 * np.pi * 14  # angular velocity of bending (rad/s): 2 * pi * freq
        """ state
        base phase: phi = -2 * pi * psi * (i - 1), where `i` is the body index (1~12)
        """
        self.phi = -2 * np.pi * self.psi * np.arange(0, self.n - 1)

    def _backward(self, step_0):
        """ phase for backward movement
        Args:
            step_0: if not None, move in the other direction
        move in the other direction
            phi = pi + 2 * omega * t_0 - phi, where t_0 is initiation time
        """
        if step_0 is not None:
            phi = np.pi + 2 * self.omega * step_0 * self.dt - self.phi
        else:
            phi = self.phi
        return phi

    def _joint_angle(self, step, step_0=None):
        """ calculate joint angles
        movement direction
            backward: q = q_max * sin(omega * t - phi)
            forward: q = q_max * sin(omega * t - (pi - phi))
        """
        phi = self._backward(step_0=step_0)
        q = self.q_max * np.sin(self.omega * step * self.dt - (np.pi - phi))
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
