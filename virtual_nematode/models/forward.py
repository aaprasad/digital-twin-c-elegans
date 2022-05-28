""" forward sinusoidal movement based on joint angle function
https://doi.org/10.1038/s41598-018-35157-1
"""

import numpy as np


class Forward(object):
    """ generate action to perform forward sinusoidal movement based on joint angle function """
    def __init__(self, dt, seed=None, n=12, q_max=40, a_max=2., psi=0.2, freq=3., motor=True):
        self.dt = dt  # real time per step
        self.n = n  # number of bodies
        self.q_max = q_max * np.pi / 180  # max joint angle (rad)
        self.motor = motor  # if True, return motor control, else return position control
        """ affect sinusoidal posture and speed """
        self.a_max = a_max  # action: [-a_max, a_max], only applicable when motor=True
        self.psi = psi  # body wavelength (rad)
        self.omega = 2 * np.pi * freq  # angular velocity of bending (rad/s): 2 * pi * freq
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
        if self.motor is True:  # return motor control
            action = self._action(q=q, q_next=q_next, q_vel=q_vel)
        else:  # return position control
            action = q_next
        return action

    def stimuli(self, step):
        """ external stimulus signal used as dataset input
        sine_wave or square_wave
        """
        q = self._joint_angle(step=step)  # current step's joint angles
        q = q[0]  # first joint's angle
        return q


class ForwardZY(Forward):
    """ generate action around z- and y-axis to perform forward sinusoidal movement based on joint angle function
    action space: Box(-100.0, 100.0, (48,), float32)
        [0:24]: angles applied on the position servos around z-axis (angle, rad)
        [24:48]: angles applied on the position servos around y-axis (angle, rad)
    """
    def __init__(self, y_ctrl, **kwargs):
        super(ForwardZY, self).__init__(**kwargs)
        self.y_ctrl = y_ctrl

    def step(self, step, q, q_vel):
        action = super(ForwardZY, self).step(step, q, q_vel)
        action = np.concatenate((action, self.y_ctrl))
        return action


class ForwardMuscle(object):
    def __init__(self, dt, n, a, freq, psi, kp, kv):
        self.dt = dt  # real time per step
        self.n = n  # number of bodies
        self.a = a  # max control value
        self.omega = 2 * np.pi * freq
        self.psi = psi  # body wavelength (rad)
        self.phi = -2 * np.pi * self.psi * np.arange(0, self.n - 1) - np.pi
        self.kp = kp  # position feedback gain
        self.kv = kv  # velocity feedback gain

    def step(self, step, q, q_vel, **kwargs):
        """ get next step's angles as control signal """
        step += 1
        q_next = self.a * np.sin(self.omega * step * self.dt + self.phi)
        action = (q_next - q) * self.kp - q_vel * self.kv
        return action
