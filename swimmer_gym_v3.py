"""
Swimmer-v3 from OpenAI Gym
"""

import gym
import os
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder


def test_random():
    # make env
    env = gym.make('Swimmer-v3')

    # action and observation space
    # print(env.action_space, env.action_space.low, env.action_space.high)
    # print(env.observation_space, env.observation_space.low, env.observation_space.high)

    # record video
    os.makedirs('video', exist_ok=True)
    rec = VideoRecorder(env, base_path='video/swimmer_gym_v3', enabled=True)

    # run and record
    observation = env.reset()
    rec.capture_frame()
    for i in range(1000):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rec.capture_frame()
        if done:
            print("Episode finished after {} steps".format(i + 1))
            break
    rec.close()
    env.close()


def train_stable_baselines3(train: bool):
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3 import A2C, DDPG, HER, PPO, SAC, TD3
    from stable_baselines3.common.env_util import make_vec_env

    """ make and check env """
    env = gym.make('Swimmer-v3')  # Swimmer-v2, Swimmer-v3 (fixed 1000 steps each episode)
    # env = make_vec_env('Swimmer-v3', n_envs=4)  # Parallel environments
    # check_env(env)

    """ build RL model:
    A2C, DDPG, HER, PPO, SAC, TD3
    """
    model = DDPG('MlpPolicy', env, verbose=1, device=torch.device('cuda:2'))  # train 1500000 steps, avg reward over 100 consecutive episodes: 13.6537
    # model = A2C('MlpPolicy', env, verbose=1)  # avg reward over 100 consecutive episodes: 31.1302, 31.1832, 31.0867
    # model = PPO('MlpPolicy', env, verbose=1)  # avg reward over 10 consecutive episodes: 23.5533
    # model = SAC('MlpPolicy', env, verbose=1)  # avg reward over 100 consecutive episodes: 27.1014, 28.3048, 28.9001
    # model = TD3('MlpPolicy', env, verbose=1)  # avg reward over 10 consecutive episodes: 18.0657

    """ train, save and load model """
    if train is True:
        os.makedirs('checkpoint', exist_ok=True)
        model.learn(total_timesteps=1500000)  # DDPG: 1500000 (OpenAI Spinning Up's PyTorch benchmarks), others: 10000
        model.save('checkpoint/swimmer_gym_v3_ddpg')
    else:
        model = DDPG.load('checkpoint/swimmer_gym_v3_ddpg')

    return env, model


def test_stable_baselines3(train: bool):
    # define env, train, save and load model
    env, model = train_stable_baselines3(train=train)

    # test RL model: avg reward over 100 consecutive episodes
    total_reward_list = []
    for e in range(100):
        print(e)
        observation = env.reset()
        total_reward = 0.
        for i in range(1000):  # register: max_episode_steps=1000
            # env.render()
            action, state = model.predict(observation=observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:  # if done.any():
                print("Episode finished after {} steps".format(i + 1))
                break
        print('Episode {} reward: {}'.format(e, total_reward))
        total_reward_list.append(total_reward)
    env.close()
    print('{} episodes mean reward: {}'.format(len(total_reward_list), sum(total_reward_list) / len(total_reward_list)))


def train_garage(train: bool):
    """ Train TRPO with Swimmer-v3 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    Reference:
        https://github.com/rlworkgroup/garage/blob/master/src/garage/examples/torch/trpo_pendulum.py
    """
    from garage import wrap_experiment
    from garage.envs import GymEnv
    from garage.experiment import Snapshotter
    from garage.experiment.deterministic import set_seed
    from garage.sampler import LocalSampler
    from garage.torch.algos import TRPO
    from garage.torch.policies import GaussianMLPPolicy
    from garage.torch.value_functions import GaussianMLPValueFunction
    from garage.trainer import Trainer
    import os
    import time

    # set log_dir
    timestamp = time.time()
    date = time.strftime('%Y%m%d-%H%M%S', time.localtime(timestamp))
    log_dir = os.path.join('log', date)
    # set save_dir
    os.makedirs('checkpoint', exist_ok=True)
    save_dir = 'checkpoint/swimmer_gym_v3_trpo'

    @wrap_experiment(log_dir=log_dir, snapshot_mode='all')
    def train_wrapper(ctxt=None, seed=1):
        # set seed
        set_seed(seed)
        # make env
        env = GymEnv('Swimmer-v3')
        # build RL model
        trainer = Trainer(ctxt)
        policy = GaussianMLPPolicy(
            env.spec, hidden_sizes=[32, 32], hidden_nonlinearity=torch.tanh, output_nonlinearity=None
        )
        value_function = GaussianMLPValueFunction(
            env_spec=env.spec, hidden_sizes=(32, 32), hidden_nonlinearity=torch.tanh, output_nonlinearity=None
        )
        sampler = LocalSampler(agents=policy, envs=env, max_episode_length=env.spec.max_episode_length)
        algo = TRPO(
            env_spec=env.spec, policy=policy, value_function=value_function, sampler=sampler, discount=0.99,
            center_adv=False
        )
        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=1024)

        # save model
        trainer.save(save_dir)
        return env, trainer

    def load_wrapper():
        # Load the env and policy from snap-shot
        snapshotter = Snapshotter()
        data = snapshotter.load(save_dir)
        env = data['env']
        policy = data['algo'].policy
        return env, policy

    if train is True:
        return train_wrapper()
    else:
        return load_wrapper()


def test_garage(train: bool):
    env, policy = train_garage(train=train)

    total_reward_list = []
    for e in range(100):
        print(e)
        observation = env.reset()
        total_reward = 0.
        for i in range(1000):  # register: max_episode_steps=1000
            # env.render()
            action = policy.get_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print("Episode finished after {} steps".format(i + 1))
                break
        print('Episode {} reward: {}'.format(e, total_reward))
        total_reward_list.append(total_reward)
    env.close()
    print('{} episodes mean reward: {}'.format(len(total_reward_list), sum(total_reward_list) / len(total_reward_list)))


if __name__ == '__main__':
    """ use register() to add a new environment, starts with -v0 """
    # print(gym.envs.registry.all())

    # take random actions and record video
    # test_random()

    # run RL algos from stable_baselines3
    # test_stable_baselines3(train=True)
    # test_stable_baselines3(train=False)

    # run RL algos from garage
    test_garage(train=True)
    # test_garage(train=False)
