import gym


class Distribution(gym.Wrapper):
    """ environmental substance distribution wrapper """
    def __init__(self, env, f, source):
        super(Distribution, self).__init__(env)
        self.f = f  # concentration function
        self.source = source  # source position

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, info = self.reward(reward, info)
        return observation, reward, done, info

    def reward(self, reward, info):
        position = self.env.sim.data.qpos[0:2].copy()
        concentration = self.f(position, source=self.source)
        # rewrite: reward, info
        reward = concentration
        info['concentration'] = concentration
        info['reward_concentration'] = concentration
        return reward, info
