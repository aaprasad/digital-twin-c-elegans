""" AntBulletEnv-v0 from PyBullet """

import gym


if __name__ == '__main__':
    """ register and make env """
    env = gym.make('AntBulletEnv-v0')
    # env.render(mode='human')  # needed for display

    """ record video
    - there's a relevant bug in gym==0.18.0, use 'pip install -e .' to install dev version instead
    - video will be named: based_path + '.mp4'
    """
    env = gym.wrappers.Monitor(env, directory='video/ant_bullet_v0', force=True)

    """ run and record """
    observation = env.reset()
    # print(observation)
    for i in range(1000):
        # env.render(mode='human')  # show the current frame of visualization
        action = env.action_space.sample()  # sample a random action
        observation, reward, done, info = env.step(action)
        # print(observation)
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    env.close()
