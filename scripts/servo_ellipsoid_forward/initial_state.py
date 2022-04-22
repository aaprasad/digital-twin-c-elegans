""" get y-axis rotation angles because of gravity """

import csv
import numpy as np
from virtual_nematode.envs.ellipsoid import make_swimmer_y


if __name__ == '__main__':
    env = make_swimmer_y(n_bodies=25, joint_range='-100 100', max_episode_steps=1000, reset_noise_scale=0.)
    print(env.action_space)
    print(env.observation_space)
    # simulate with zero z-axis rotation ctrl
    observation = env.reset()
    for i in range(10 ** 6):
        # env.render()
        action = np.zeros_like(env.action_space)
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} steps'.format(i + 1))
            break
    env.close()
    # y-axis rotation because of gravity
    observation = observation.astype(np.float32)
    angles = observation[4:28]
    print(angles)
    with open('y_axis_angles_32bit.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(np.expand_dims(angles, axis=0).T)
