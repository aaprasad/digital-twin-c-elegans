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
        self.phi = 2 * np.pi * self.psi * np.arange(0, self.n - 1)

    def reset(self):
        pass

    def step(self, step, **kwargs):
        """ get next step's angles as control signal """
        step += 1
        q_next = self.a * np.sin(self.omega * step * self.dt + self.phi)
        return q_next

    def stimuli(self, step):
        """ extra signal used as dataset input """
        return None


class SinusoidalServo(Sinusoidal):
    """ generate sinusoidal servo control signal """
    def __init__(self, y_angle, **kwargs):
        super(SinusoidalServo, self).__init__(**kwargs)
        self.y_angle = y_angle  # joint angles around y-axis (angle, rad)

    def step(self, step, **kwargs):
        action = super(SinusoidalServo, self).step(step, **kwargs)
        action = np.concatenate((action, self.y_angle))
        return action


class SinusoidalMuscle(Sinusoidal):
    def step(self, step, **kwargs):
        """ ellipsoid swimmer with tendon muscles
        action space: Box(0.0, 1.0, (95,), float32)
            [0:24]: control signal for DL quadrant's 24 muscles
            [24:48]: DR's 24
            [48:71]: VL's 23
            [71:95]: VR's 24
        """
        action = super(SinusoidalMuscle, self).step(step, **kwargs)
        dorsal = (action <= 0.) * np.abs(action)
        ventral = (action >= 0.) * action
        action = np.concatenate((dorsal, dorsal, ventral[0:23], ventral), axis=0)
        return action
