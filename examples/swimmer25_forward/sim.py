""" swimmer: forward locomotion
swimmer configuration
    joint size = 0.1 <- radius = 0.04mm
    whole body len = 2.5 <- length = 1mm
    joint amount = 24 <- amount of muscles in one quadrant = 24
    n_bodies = 25
    body len = 0.1
observation space: (52,) including qpos[2:27], qvel[0:25],
    qpos[0:27]: x_pos, y_pos, ?, q[0:24]
    qvel[0:27]: x_vel, y_vel, ?, q_vel[0:24]
"""

import numpy as np
from virtual_nematode.envs.swimmer_v3_v2 import make_swimmer
from virtual_nematode.models.forward import Forward


def simulate(env, seed=None, max_episode_steps=2500, trials=1):
    """ forward sinusoidal movement """
    displacements = []
    if trials > 1:
        seed = None  # ensure that seed is different in each trial
    for i in range(trials):
        env.seed(seed)
        observation = env.reset()
        model = Forward(dt=env.dt, seed=seed, n=25)
        for step in range(10 ** 6):
            # env.render()
            action = model.step(step=step, q=observation[1:25], q_vel=observation[28:])
            observation, reward, done, info = env.step(action)
            if done:
                d = np.linalg.norm(np.array(env.stats['com'][-1]) - np.array(env.stats['com'][0]), ord=2)
                displacements.append(d)
                print('Trial {}: com displacement {:.2f} / {} steps'.format(i + 1, d, step + 1))
                break
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(displacements), np.mean(displacements), max_episode_steps))


if __name__ == '__main__':
    n_bodies = 25
    joint_range = '-100 100'
    body_len = 0.1
    camera_pos = '0 -6 6'
    camera_z = 50
    camera_name = None
    max_episode_steps = 2500
    video_name = 'swimmer'
    reset_noise_scale = 0.1
    env = make_swimmer(
        n_bodies, joint_range, body_len, camera_pos, camera_z, camera_name, max_episode_steps, video_name,
        reset_noise_scale
    )
    # print(env.action_space)
    # print(env.observation_space)
    simulate(env, seed=None, max_episode_steps=max_episode_steps, trials=1)
