from virtual_nematode.envs.muscle_ellipsoid2d import make_swimmer


if __name__ == '__main__':
    env = make_swimmer(
        n_bodies=25, joint_range=['-70 70'] + ['-100 100'] * 22 + ['-70 70'],
        max_episode_steps=2500, reset_noise_scale=0.6,
        density=1.2, viscosity=0.1, condim=3, friction='1 1 0.005 0.0001 0.0001', cone='elliptic'
    )
    print(env.action_space)
    print(env.observation_space)
