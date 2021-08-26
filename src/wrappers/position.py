import gym


class Position(gym.Wrapper):
    """ get info on position and center of mass """
    def step(self, action):
        """ override gym.Wrapper """
        observation, reward, done, info = self.env.step(action)
        position = self.env.sim.data.qpos[0:2].copy()  # position of anterior tip
        com = self.env.sim.data.subtree_com[1][0:2].copy()  # center of mass
        info['position'] = position.tolist()
        info['com'] = com.tolist()
        return observation, reward, done, info
