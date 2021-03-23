"""
Swimmer-v3 from OpenAI Gym
"""

import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, DDPG, HER, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env


def get_model_stable_baselines():
    """ build RL model:
    A2C, DDPG, HER, PPO, SAC, TD3
    """
    # model = A2C('MlpPolicy', env, verbose=1)  # avg reward over 100 consecutive episodes: 31.1302, 31.1832, 31.0867
    model = DDPG(
        'MlpPolicy', env, verbose=1, device=torch.device('cuda:2')
    )  # avg reward over 10 consecutive episodes: 11.0541
    # model = PPO('MlpPolicy', env, verbose=1)  # avg reward over 10 consecutive episodes: 23.5533
    # model = SAC('MlpPolicy', env, verbose=1)  # avg reward over 100 consecutive episodes: 27.1014, 28.3048, 28.9001
    # model = TD3('MlpPolicy', env, verbose=1)  # avg reward over 10 consecutive episodes: 18.0657

    """ train, save and load model """
    model.learn(total_timesteps=1500000)
    model.save('Swimmer_v3_DDPG')
    # model = DDPG.load('Swimmer_v3_DDPG')

    return model


if __name__ == '__main__':
    """ use register() to add a new environment, starts with -v0 """
    # print(gym.envs.registry.all())

    """ make and check env """
    env = gym.make('Swimmer-v3')  # Swimmer-v2, Swimmer-v3 (fixed 1000 steps each episode)
    # env = make_vec_env('Swimmer-v3', n_envs=4)  # Parallel environments
    # check_env(env)

    """ action and observation space """
    # print(env.action_space, env.action_space.low, env.action_space.high)
    # print(env.observation_space, env.observation_space.low, env.observation_space.high)

    """ define, train, save and load model """
    model = get_model_stable_baselines()

    """ test RL model
    - avg reward over 100 consecutive episodes
    """
    total_reward_list = []
    for e in range(100):
        print(e)
        observation = env.reset()
        # print(observation)
        total_reward = 0.
        for i in range(1000):  # register: max_episode_steps=1000
            # env.render()  # show the current frame of visualization
            # action = env.action_space.sample()  # sample a random action
            action, state = model.predict(observation=observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            # print(env.sim.data.qpos, observation)
            # print(observation)
            if done:  # if done.any():
                print("Episode finished after {} steps".format(i + 1))
                break
        print('Episode {} reward: {}'.format(e, total_reward))
        total_reward_list.append(total_reward)
    env.close()
    print('{} episodes mean reward: {}'.format(len(total_reward_list), sum(total_reward_list) / len(total_reward_list)))
