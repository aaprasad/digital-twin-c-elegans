""" get framepos and framequat because of gravity """

import json
import numpy as np
from virtual_nematode.envs.muscle_ellipsoid import make_swimmer_y_with_frame_sensor


if __name__ == '__main__':
    # create env
    env = make_swimmer_y_with_frame_sensor(
        n_bodies=25, joint_range='-100 100', y_joint_ranges=['-100 100'] * 24, theta=np.pi / 12, max_episode_steps=2000,
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
    sensordata = observation[254:]
    # data processing
    data = {
        # body related sensor data
        'body_framepos': [], 'body_framequat': [], 'joint_geom_framepos': [],
        # muscle related sensor data
        'sidesite_dorsal_framepos': [], 'sidesite_ventral_framepos': [],
        'site_anterior_dbwml': [], 'site_anterior_dbwmr': [], 'site_anterior_vbwml': [], 'site_anterior_vbwmr': [],
        'site_posterior_dbwml': [], 'site_posterior_dbwmr': [], 'site_posterior_vbwml': [], 'site_posterior_vbwmr': [],
        # body related extra data
        'body_framepos_ref': [], 'joint_geom_framepos_ref': [],
        # muscle related sensor data
        'sidesite_dorsal_framepos_ref': [], 'sidesite_ventral_framepos_ref': [],
        'site_anterior_dbwml_ref': [], 'site_anterior_dbwmr_ref': [], 'site_anterior_vbwml_ref': [], 'site_anterior_vbwmr_ref': [],
        'site_posterior_dbwml_ref': [], 'site_posterior_dbwmr_ref': [], 'site_posterior_vbwml_ref': [], 'site_posterior_vbwmr_ref': [],
    }
    # body related sensor data
    for i in range(25):
        data['body_framepos'].append(sensordata[i * 3: 3 + i * 3])
        data['body_framequat'].append(sensordata[75 + i * 4: 79 + i * 4])
    for i in range(24):
        data['joint_geom_framepos'].append(sensordata[175 + i * 3: 178 + i * 3])
    # muscle related sensor data
    for i in range(24):
        data['sidesite_dorsal_framepos'].append(sensordata[247 + i * 3: 250 + i * 3])
        data['sidesite_ventral_framepos'].append(sensordata[319 + i * 3: 322 + i * 3])
        data['site_anterior_dbwml'].append(sensordata[391 + i * 3: 394 + i * 3])
        data['site_anterior_dbwmr'].append(sensordata[463 + i * 3: 466 + i * 3])
        data['site_anterior_vbwml'].append(sensordata[535 + i * 3: 538 + i * 3])
        data['site_anterior_vbwmr'].append(sensordata[607 + i * 3: 610 + i * 3])
        data['site_posterior_dbwml'].append(sensordata[679 + i * 3: 682 + i * 3])
        data['site_posterior_dbwmr'].append(sensordata[751 + i * 3: 754 + i * 3])
        data['site_posterior_vbwml'].append(sensordata[823 + i * 3: 826 + i * 3])
        data['site_posterior_vbwmr'].append(sensordata[895 + i * 3: 898 + i * 3])
    # body related extra data
    for i in range(25):
        if i == 0:
            pos = data['body_framepos'][i]
        else:
            pos = data['body_framepos'][i] - data['body_framepos'][i - 1]
        data['body_framepos_ref'].append(pos)
    for i in range(24):
        refpos = data['joint_geom_framepos'][i] - data['body_framepos'][i + 1]
        data['joint_geom_framepos_ref'].append(refpos)
    # muscle related sensor data
    for key in data:
        if key.endswith('ref') is False:
            key_ref = '{}_ref'.format(key)
            if key.startswith('sidesite') or key.startswith('site_posterior'):
                data[key_ref] = [data[key][i] - data['body_framepos'][i + 1] for i in range(24)]
            elif key.startswith('site_anterior'):
                data[key_ref] = [data[key][i] - data['body_framepos'][i] for i in range(24)]
    # to list
    for key in data:
        for i in range(len(data[key])):
            data[key][i] = data[key][i].tolist()
    print(data)
    with open('frame_sensor_for_muscle.json', 'w') as f:
        json.dump(data, f, indent=4)
