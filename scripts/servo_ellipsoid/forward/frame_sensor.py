""" get framepos and framequat because of gravity """

import json
import numpy as np
from virtual_nematode.envs.ellipsoid import make_swimmer_y_with_frame_sensor


if __name__ == '__main__':
    # create env
    env = make_swimmer_y_with_frame_sensor(
        n_bodies=25, joint_range='-100 100', y_joint_ranges=['-100 100'] * 24, max_episode_steps=1000,
        reset_noise_scale=0., density=4000, viscosity=0.1, condim=3, friction='1 1'
    )
    print(env.action_space)
    print(env.observation_space)
    # simulate with zero ctrl
    observation = env.reset()
    for i in range(10 ** 6):
        # env.render()
        action = np.zeros_like(env.action_space)
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} steps'.format(i + 1))
            break
    env.close()
    # final observation
    observation = observation.astype(np.float16)
    sensordata = observation[62:]
    # data processing
    data = {
        # sensor data
        'body_framepos': [], 'body_framequat': [], 'joint_geom_framepos': [],
        # extra data
        'body_framepos_ref': [], 'joint_geom_framepos_ref': []
    }
    # sensor data
    for i in range(25):
        data['body_framepos'].append(sensordata[i * 3: 3 + i * 3])
        data['body_framequat'].append(sensordata[75 + i * 4: 79 + i * 4])
    for i in range(24):
        data['joint_geom_framepos'].append(sensordata[175 + i * 3: 178 + i * 3])
    # extra data
    for i in range(25):
        if i == 0:
            pos = data['body_framepos'][i]
        else:
            pos = data['body_framepos'][i] - data['body_framepos'][i - 1]
        data['body_framepos_ref'].append(pos)
    for i in range(24):
        refpos = data['joint_geom_framepos'][i] - data['body_framepos'][i + 1]
        data['joint_geom_framepos_ref'].append(refpos)
    # to list
    for key in data:
        for i in range(len(data[key])):
            data[key][i] = data[key][i].tolist()
    print(data)
    with open('frame_sensor.json', 'w') as f:
        json.dump(data, f, indent=4)
