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
        self.step_b0 = None  # start of backward movement
        self.step_omega0 = None  # start of sharp turn
        """ backward """
        self.step_b = 100  # backward movement
        """ sharp turn """
        self.step_omega1 = 100  # sharp turn phase 1
        self.step_omega2 = 100  # sharp turn phase 2
        self.step_omega3 = self.step_omega1  # sharp turn phase 3
        self.c_omega = 50  # phase delay for changing posture from S-shaped to omega-shaped and back
        self.kappa_omega = -0.35  # bias angle
        """ weathervane
        c_w: weathervane curvature coefficient
            needs to be large enough to expand gradient range to tanh's discriminative range ~[-3, 3]
            related to gaussian distribution's sigma, which in effect defines the range where gradient exists
            larger: weathervane can work in longer distance, but not as precise in close range
            smaller: weathervane only works in smaller distance, but more precise
        kappa_w_max: max weathervane bias angle
            define range of tanh() to limit bias angle which prevents sharp turn
        """
        self.c_w = 100
        self.kappa_w_max = 0.3

    def _backward(self, step):
        """ update phase for backward movement
        apply once at the beginning to move in the other direction
            phi = pi + 2 * omega * t_0 - phi, where t_0 is initiation time
        apply it again at the end to switch back to original direction
        """
        if self.step_b0 is not None and (step == self.step_b0 or step == self.step_b0 + self.step_b):
            self.phi = np.pi + 2 * self.omega * self.step_b0 * self.dt - self.phi

    def _sharp_turn(self, step):
        """ update phase for sharp turn
        it changes posture but still needs bias angle param to perform a sharp turn
        phase 1: phi(t) = phi(t - dt) + c_omega / t_omega1 * dt, where t_omega1 is duration
        phase 2: phi(t) = phi(t), where t_omega2 is duration
        phase 3: phi(t) = phi(t - dt) - c_omega / t_omega3 * dt, where t_omega3 is duration
        return bias angle for performing sharp turn
        """
        if self.step_omega0 is not None:
            phase0 = self.step_omega0
            phase1 = phase0 + self.step_omega1
            phase2 = phase1 + self.step_omega2
            phase3 = phase2 + self.step_omega3
            if phase0 <= step < phase1:
                self.phi += self.c_omega / self.step_omega1
            elif phase2 <= step < phase3:
                self.phi -= self.c_omega / self.step_omega3
            if phase0 <= step < phase3:
                return self.kappa_omega
        return 0.

    def _pirouette(self, step):
        """ set up pirouette start time
        phase 1: backward movement
        phase 2: sharp turn
        return True if it's performing pirouette, False if it's not
        """
        if self.step_b0 is None:  # not performing pirouette
            if step == 300:  # initiate pirouette
                self.step_b0 = step
                self.step_omega0 = step + self.step_b
        elif step >= self.step_b0 + self.step_b + self.step_omega1 + self.step_omega2 + self.step_omega3:  # pirouette has finished
            self.step_b0 = None
            self.step_omega0 = None

    def _weathervane(self, g_w):
        """ use normal gradient to get weathervane bias angle
        original function
            kappa_w = -c_w * g_w
            In gaussian distribution, it only works in close proximity because distant gradient is way too small
        tanh: prevent sharp turn
        """
        kappa_w = -self.c_w * g_w
        kappa_w = np.tanh(kappa_w) * self.kappa_w_max
        return kappa_w

    def _joint_angle(self, step, g_w):
        """ calculate joint angles
        movement direction
            backward: q = q_max * sin(omega * t - phi)
            forward: q = q_max * sin(omega * t - (pi - phi))
        """
        self._backward(step=step)
        kappa_omega = self._sharp_turn(step=step)
        kappa_w = self._weathervane(g_w=g_w)
        q = self.q_max * np.sin(self.omega * step * self.dt - (np.pi - self.phi)) + kappa_omega + kappa_w
        return q

    def _action(self, q, q_next, q_vel):
        """ delta_q: [-4 * q_max, 4 * q_max] -> action: [-a_max, a_max]
        - consider (q_vel_next - q_vel) for action
        """
        delta_q = q_next - q - q_vel * self.dt
        action = delta_q / (4 * self.q_max) * self.a_max
        return action

    def step(self, step, q, q_vel, g_w):
        self._pirouette(step=step)
        q_next = self._joint_angle(step=step, g_w=g_w)
        action = self._action(q=q, q_next=q_next, q_vel=q_vel)
        return action
