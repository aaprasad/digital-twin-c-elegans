""" swimmer: forward locomotion """

from virtual_nematode.envs.swimmer_forward import simulate


if __name__ == '__main__':
    simulate(max_episode_steps=2500, seed=None, trials=100)
