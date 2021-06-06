""" work in progress """

""" Swimmer-Gym-v3-Bullet-v0 is modeled after Swimmer-v3 from OpenAI Gym
- with Bullet physics instead of MuJoCo
- tries to be the same as Swimmer-v3
"""

import gym
# import pybullet
import pybullet_envs
from gym.envs.registration import register

"""
if __name__ == '__main__':
    register(
        id='Swimmer-Gym-v3-Bullet-v0',
        entry_point='src.envs.bullet.swimmer_gym_v3_bullet_v0:SwimmerEnv',
        max_episode_steps=1000,
        reward_threshold=360.0,
    )
    env = gym.make('Swimmer-Gym-v3-Bullet-v0')
    env.render(mode='human')
    # env = gym.wrappers.Monitor(env, directory='video/swimmer_gym_v3_bullet_v0', force=True)

    observation = env.reset()
    for i in range(1000):
        env.render(mode='human')
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    env.close()
"""
