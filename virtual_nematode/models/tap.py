""" tap-withdrawal composing of forward, backward and stochastic turning behaviors """

import numpy as np


class Tap(object):
    """ tap-withdrawal
    forward: default behavior
    tap head/tail: reverse for a short period, then stochastic forward turning, and then forward
    """
    def __init__(self, dt, seed=None, n=12, q_max=40, a_max=2., psi=0.2, freq=3.):
        self.dt = dt  # real time per step
        self.n = n  # number of bodies
        self.q_max = q_max * np.pi / 180  # max joint angle (rad)
        """ affect sinusoidal posture and speed """
        self.a_max = a_max  # action: [-a_max, a_max]
        self.psi = psi  # body wavelength (rad)
        self.omega = 2 * np.pi * freq  # angular velocity of bending (rad/s): 2 * pi * freq
        self.step_b = 100  # reverse duration
        self.step_w = 100  # turning duration
        self.kappa_w_max = 0.3  # turning angle range
        """ state """
        self.phi = -2 * np.pi * self.psi * np.arange(0, self.n - 1)
        self.step_b0 = None
        self.kappa_w = None
        """ seeding """
        self.seed(seed)

    @staticmethod
    def seed(seed=None):
        if seed is not None:
            np.random.seed(seed)

    def _backward(self, step):
        """ reverse """
        if self.step_b0 is not None and (step == self.step_b0 or step == self.step_b0 + self.step_b):
            self.phi = np.pi + 2 * self.omega * self.step_b0 * self.dt - self.phi

    def _weathervane(self, step):
        """ sample random weathervane bias angle """
        if self.step_b0 is not None and self.step_b0 + self.step_b <= step < self.step_b0 + self.step_b + self.step_w:
            kappa_w = self.kappa_w
        else:
            kappa_w = 0.
        return kappa_w

    def _joint_angle(self, step):
        """ calculate joint angles """
        self._backward(step=step)
        kappa_w = self._weathervane(step)
        q = self.q_max * np.sin(self.omega * step * self.dt - (np.pi - self.phi)) + kappa_w
        return q

    def _action(self, q, q_next, q_vel):
        delta_q = q_next - q - q_vel * self.dt
        action = delta_q / (4 * self.q_max) * self.a_max
        return action

    def step(self, step, q, q_vel, force):
        if force[0] and self.step_b0 is None:
            self.step_b0 = step + 1  # start reverse phase next step
            self.kappa_w = np.random.uniform(-self.kappa_w_max, self.kappa_w_max)
        elif self.step_b0 is not None and step >= self.step_b0 + self.step_b + self.step_w:
            self.step_b0 = None  # both reverse and turning phase has finished
            self.kappa_w = None
        q_next = self._joint_angle(step=step + 1)  # next step's joint angles
        action = self._action(q=q, q_next=q_next, q_vel=q_vel)
        return action

    def stimuli(self, step):
        """ external stimulus signal used as dataset input
        sine_wave or square_wave
        """
        q = self._joint_angle(step=step)  # current step's joint angles
        q = q[0]  # first joint's angle
        return q
