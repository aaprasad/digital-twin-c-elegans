""" forward sinusoidal movement
https://doi.org/10.1038/s41598-018-35157-1
"""

import numpy as np


class Forward(object):
    """ generate action to perform forward sinusoidal movement """
    def __init__(self, dt, seed=None, n=12):
        self.dt = dt  # real time per step
        self.n = n  # number of bodies
        self.q_max = 40 * np.pi / 180  # max joint angle (rad)
        """ affect sinusoidal posture and speed """
        self.a_max = 2.  # action: [-a_max, a_max]
        self.psi = 0.2  # body wavelength (rad)
        self.omega = 2 * np.pi * 3  # angular velocity of bending (rad/s): 2 * pi * freq
        """ state """
        self.phi = -2 * np.pi * self.psi * np.arange(0, self.n - 1)
        """ seeding """
        self.seed(seed)

    @staticmethod
    def seed(seed=None):
        if seed is not None:
            np.random.seed(seed)

    def _joint_angle(self, step):
        """ calculate joint angles
        forward:
            q = q_max * sin(omega * t - (pi - phi))
            phi = -2 * pi * psi * (i - 1), where `i` is the body index (1~12)
        """
        q = self.q_max * np.sin(self.omega * step * self.dt - (np.pi - self.phi))
        return q

    def _action(self, q, q_next, q_vel):
        """ delta_q: [-4 * q_max, 4 * q_max] -> action: [-a_max, a_max]
        - consider (q_vel_next - q_vel) for action
        """
        delta_q = q_next - q - q_vel * self.dt
        action = delta_q / (4 * self.q_max) * self.a_max
        return action

    def step(self, step, q, q_vel):
        q_next = self._joint_angle(step=step + 1)  # next step's joint angles
        action = self._action(q=q, q_next=q_next, q_vel=q_vel)
        return action

    def stimuli(self, step, mode):
        """ external stimulus signal used as dataset input
        mode: 'sine_wave' or 'square_wave'
        """
        q = self._joint_angle(step=step)  # current step's joint angles
        q = q[0]  # first joint's angle
        if mode == 'sine_wave':
            pass
        elif mode == 'square_wave':
            q = np.sign(q)
        else:
            raise AssertionError('Unidentified mode {}.'.format(mode))
        return q
