import numpy as np


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
        dorsal = (action <= 0.) * np.abs(action)
        ventral = (action >= 0.) * action
        action = np.concatenate((dorsal, dorsal, ventral[0:23], ventral), axis=0)
        return action


class ForwardPIDMuscle(object):
    """ PD controller """
    def __init__(self, dt, n, a, freq, psi, kp, kd):
        self.dt = dt  # real time per step
        self.n = n  # number of bodies
        self.a = a  # max control value
        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.psi = psi  # body wavelength (rad)
        self.phi = -2 * np.pi * self.psi * np.arange(0, self.n - 1) - np.pi
        self.kp = kp  # position gain
        self.kd = kd  # derivative gain
        # self.kp_direction = kp_direction  # gradual turn coefficient
        # state
        self.last_error = None
        self.start_step = None
        # self.direction_target = None  # np.array([1, 0])

    def seed(self, **kwargs):
        pass

    def reset(self):
        self.last_error = 0.
        self.start_step = 0
        # self.start_step = np.random.randint(0, np.ceil(1. / (self.freq * self.dt)))
        # self.direction_target = None

    def _action(self, q, q_target):
        error = q_target - q
        u = self.kp * error + self.kd * (error - self.last_error) / self.dt  # + self.kp_direction * theta_error
        dorsal = (u <= 0.) * np.abs(u)
        ventral = (u >= 0.) * u
        # ventral[23] *= 2
        action = np.concatenate((dorsal, dorsal, ventral[0:23], ventral), axis=0)
        self.last_error = error
        return action

    def step(self, step, q, **kwargs):
        # if self.direction_target is None:
        #     self.direction_target = direction
        # theta_error = np.arctan2(
        #     self.direction_target[0] * direction[1] - self.direction_target[1] * direction[0],
        #     self.direction_target[0] * direction[0] + self.direction_target[1] * direction[1]
        # )  # direction turns theta clockwise to target direction (-pi~pi, rad)
        q_target = self.a * np.sin(self.omega * (step + self.start_step) * self.dt + self.phi)  # + self.kp_direction * theta_error
        action = self._action(q, q_target)
        return action


class ReversePIDMuscle(ForwardPIDMuscle):
    def _reverse(self, step, reverse):
        if reverse is True:
            self.phi = 2 * self.omega * step * self.dt - self.phi

    def step(self, step, q, **kwargs):
        self._reverse(step, reverse=kwargs.get('reverse'))
        action = super(ReversePIDMuscle, self).step(step, q, **kwargs)
        return action


class WeathervanePIDMuscle(ForwardPIDMuscle):
    def __init__(self, k_w, **kwargs):
        super(WeathervanePIDMuscle, self).__init__(**kwargs)
        self.k_w = k_w

    def step(self, step, q, **kwargs):
        g_w = kwargs.get('g_w')
        q_target = self.a * np.sin(self.omega * step * self.dt + self.phi) - self.k_w * g_w
        action = self._action(q, q_target)
        return action


class ShallowTurn(ForwardPIDMuscle):
    def __init__(self, k_w, sigma, **kwargs):
        super(ShallowTurn, self).__init__(**kwargs)
        self.k_w = k_w
        self.sigma = sigma

    def step(self, step, q, **kwargs):
        start_step = kwargs.get('start_step')
        n = np.arange(0, self.n - 1)
        # bias = np.exp(-(n - self.omega * (step - start_step) * self.dt) ** 2 / (2 * self.sigma ** 2))
        speed = 2 * np.pi * self.psi / (self.omega * self.dt)
        bias = 0.5 * np.exp(-(n - (step - start_step) / speed) ** 2 / (2 * self.sigma ** 2))
        q_target = self.a * np.sin(self.omega * step * self.dt + self.phi) - bias
        # bias = 1 - 0.99 * np.exp(-(n - (step - start_step) / speed / 5) ** 2 / (2 * self.sigma ** 2))
        # q_target = self.a * np.sin(self.omega * step * self.dt + self.phi) * bias
        action = self._action(q, q_target)
        return action


class WeathervaneMuscle(object):
    def __init__(self, dt, n, a, freq, psi, kp, kv):
        self.dt = dt  # real time per step
        self.n = n  # number of bodies
        self.a = a  # max control value
        self.omega = 2 * np.pi * freq
        self.psi = psi  # body wavelength (rad)
        self.phi = -2 * np.pi * self.psi * np.arange(0, self.n - 1) - np.pi
        self.kp = kp  # position feedback gain
        self.kv = kv  # velocity feedback gain
        self.c_w = 100
        self.kappa_w_max = 0.2

    def step(self, step, q, q_vel, g_w, **kwargs):
        """ get next step's angles as control signal """
        step += 1
        q_next = self.a * np.sin(self.omega * step * self.dt + self.phi)
        action = (q_next - q) * self.kp - q_vel * self.kv
        kappa_w = np.tanh(-self.c_w * g_w) * self.kappa_w_max
        action[0:8] += kappa_w
        dorsal = (action <= 0.) * np.abs(action)
        ventral = (action >= 0.) * action
        action = np.concatenate((dorsal, dorsal, ventral[0:23], ventral), axis=0)
        return action
