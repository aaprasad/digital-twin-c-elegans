""" swimmer: forward locomotion """

from virtual_nematode.simulation.forward import simulate


if __name__ == '__main__':
    """ results
    100 trials: com displacement mean 40.84 / 2500 steps
    """
    simulate(max_episode_steps=2500, reset_noise_scale=1.745, seed=None, trials=100)
