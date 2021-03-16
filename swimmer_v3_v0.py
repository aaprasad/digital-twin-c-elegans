import gym

from gym.envs.registration import register


if __name__ == '__main__':
    """
    Swimmer-v3-v0 is exactly the same as Swimmer-v3, with personal xml_file path configuration.
    swimmer.xml is unedited.
    """
    register(
        id='Swimmer-v3-v0',
        entry_point='src.envs.swimmer_v3_v0:SwimmerEnv',
        max_episode_steps=1000,
        reward_threshold=360.0,
    )
    env = gym.make('Swimmer-v3-v0')

    observation = env.reset()
    # print(observation)
    for i in range(10000):
        env.render()  # show the current frame of visualization
        action = env.action_space.sample()  # sample a random action
        observation, reward, done, info = env.step(action)
        # print(observation)
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    env.close()
