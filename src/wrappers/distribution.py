import gym


class Distribution(gym.Wrapper):
    """ environmental substance distribution wrapper """
    def __init__(self, env, f, source):
        super(Distribution, self).__init__(env)
        self.dt = 0.01  # time step
        self.f = f  # concentration function
        self.source = source  # source position
        """ state """
        self._com_c_past = None  # last step's concentration at the center of mass

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, info = self.reward(reward, info)
        return observation, reward, done, info

    def _tangential_gradient(self, com):
        """ gradient at the center of mass """
        com_c = self.f(com, source=self.source)
        if self._com_c_past is None:
            gradient = 0.
        else:
            gradient = (com_c - self._com_c_past) / self.dt
        self._com_c_past = com_c
        return gradient

    def reward(self, reward, info):
        position = self.env.sim.data.qpos[0:2].copy()  # position of anterior tip
        com = self.env.sim.data.subtree_com[1][0:2].copy()  # center of mass
        concentration = self.f(position, source=self.source)  # position's concentration
        tangential_gradient = self._tangential_gradient(com=com)
        # reward
        reward = concentration
        # info
        info['position'] = position.tolist()
        info['com'] = com.tolist()
        info['concentration'] = concentration
        info['tangential_gradient'] = tangential_gradient
        return reward, info
