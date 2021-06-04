import gym


class Recorder(gym.Wrapper):
    """ record statistics and override rendering method """
    def __init__(self, env, stats_name: list, camera_name=None):
        super(Recorder, self).__init__(env)
        self.stats_name = stats_name
        self.camera_name = camera_name  # None, 'track', 'fixedcam'
        """ state """
        self.stats = {name: [] for name in self.stats_name}

    def reset(self, **kwargs):
        """ override gym.Wrapper """
        for name in self.stats:
            self.stats[name] = []
        return self.env.reset(**kwargs)

    def step(self, action):
        """ override gym.Wrapper """
        observation, reward, done, info = self.env.step(action)
        for name in self.stats:
            self.stats[name].append(info[name])
        return observation, reward, done, info

    def render(self, mode='human', camera_name=None, **kwargs):
        """ override gym.Wrapper """
        if camera_name is None:
            camera_name = self.camera_name
        return self.env.render(mode, camera_name=camera_name, **kwargs)
