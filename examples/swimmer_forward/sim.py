""" swimmer: forward locomotion """

from virtual_nematode.simulation.forward import simulate


if __name__ == '__main__':
    simulate(max_episode_steps=2500, seed=None)
