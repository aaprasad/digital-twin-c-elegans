import gym
import numpy as np


class Distribution(gym.Wrapper):
    """ environmental substance distribution wrapper """
    def __init__(self, env, dt, f, source):
        super(Distribution, self).__init__(env)
        self.dt = dt  # time step
        self.delta = 0.1  # small displacement
        self.f = f  # concentration function
        self.source = source  # source position
        """ state """
        self._com_c_past = None  # last step's concentration at the center of mass
        self._com_past = None  # last step's center of mass

    def reset(self, **kwargs):
        """ override gym.Wrapper """
        self._com_c_past = None
        self._com_past = None
        return self.env.reset(**kwargs)

    def step(self, action):
        """ override gym.Wrapper """
        observation, reward, done, info = self.env.step(action)
        reward, info = self.reward(reward, info)
        return observation, reward, done, info

    def _tangential_gradient(self, com):
        """ gradient parallel to the traveling direction of the center of mass """
        com_c = self.f(com, source=self.source)
        if self._com_c_past is None:
            gradient = 0.
        else:
            gradient = (com_c - self._com_c_past) / self.dt
        self._com_c_past = com_c
        return gradient

    def _normal_gradient(self, com):
        """ gradient perpendicular to the traveling direction of the center of mass """
        if self._com_past is None:
            gradient = 0.
        else:
            # tangential direction's vector
            normal_vector = com - self._com_past
            # normal direction's unit vector
            normal_vector = np.matmul(np.array([[0, 1], [-1, 0]]), normal_vector) / np.linalg.norm(normal_vector)
            # normal vector for displacement
            normal_vector *= self.delta
            # gradient
            c1 = self.f(com + normal_vector, source=self.source)
            c2 = self.f(com - normal_vector, source=self.source)
            gradient = (c2 - c1) / (2 * self.delta)
        self._com_past = com
        return gradient

    def reward(self, reward, info):
        """ refer to gym.RewardWrapper """
        position = self.env.sim.data.qpos[0:2].copy()  # position of anterior tip
        com = self.env.sim.data.subtree_com[1][0:2].copy()  # center of mass
        concentration = self.f(position, source=self.source)  # position's concentration
        g_p = self._tangential_gradient(com=com)
        g_w = self._normal_gradient(com=com)
        # reward
        reward = concentration
        # info
        info['position'] = position.tolist()
        info['com'] = com.tolist()
        info['concentration'] = concentration
        info['g_p'] = g_p
        info['g_w'] = g_w
        return reward, info
