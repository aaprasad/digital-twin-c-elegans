""" Swimmer-v3 from OpenAI Gym """

import cv2
import gym
import torch
import numpy as np


def test_random():
    """ take random actions and record video """
    # make env
    env = gym.make('Swimmer-v3')

    # action: Box(-1.0, 1.0, (2,), float32), torque control of two joints
    print(env.action_space, env.action_space.low, env.action_space.high)
    # observation: Box(-inf, inf, (8,), float64), qpos[2:5] + qvel[0:5]
    # qpos[0:5]: x pos + y pos + ? + joint1 angle + joint2 angle
    # qvel[0:5]: x vel + y vel + ? + joint1 vel + joint2 vel
    print(env.observation_space, env.observation_space.low, env.observation_space.high)

    # record video
    env = gym.wrappers.Monitor(env, directory='video/swimmer_gym_v3', force=True)

    # run and record
    observation = env.reset()
    for i in range(1000):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} steps'.format(i + 1))
            break
    env.close()


def train_stable_baselines3(train: bool):
    """ train RL algos from stable_baselines3 """
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3 import A2C, DDPG, HER, PPO, SAC, TD3
    from stable_baselines3.common.env_util import make_vec_env

    # make and check env
    env = gym.make('Swimmer-v3')  # Swimmer-v2, Swimmer-v3 (fixed 1000 steps each episode)
    # env = make_vec_env('Swimmer-v3', n_envs=4)  # Parallel environments
    # check_env(env)

    # build RL model: A2C, DDPG, HER, PPO, SAC, TD3
    model = DDPG('MlpPolicy', env, verbose=1, device=torch.device('cuda:2'))  # train 1500000 steps, avg reward over 100 consecutive episodes: 13.6537
    # model = A2C('MlpPolicy', env, verbose=1)  # avg reward over 100 consecutive episodes: 31.1302, 31.1832, 31.0867
    # model = PPO('MlpPolicy', env, verbose=1)  # avg reward over 10 consecutive episodes: 23.5533
    # model = SAC('MlpPolicy', env, verbose=1)  # avg reward over 100 consecutive episodes: 27.1014, 28.3048, 28.9001
    # model = TD3('MlpPolicy', env, verbose=1)  # avg reward over 10 consecutive episodes: 18.0657

    # train, save and load model
    if train is True:
        model.learn(total_timesteps=1500000)  # DDPG: 1500000 (OpenAI Spinning Up's PyTorch benchmarks), others: 10000
        model.save('checkpoint/swimmer_gym_v3_ddpg')
    else:
        model = DDPG.load('checkpoint/swimmer_gym_v3_ddpg')

    return env, model


def test_stable_baselines3(train: bool):
    """ train/load and test RL algos from stable_baselines3 """
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
                print('Episode finished after {} steps'.format(i + 1))
                break
        print('Episode {} reward: {}'.format(e, total_reward))
        total_reward_list.append(total_reward)
    env.close()
    print('{} episodes mean reward: {}'.format(len(total_reward_list), np.mean(total_reward_list)))


def train_garage_torch(train: bool, log_dir: str, init_env):
    """ Train TRPO with Swimmer-v3 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce determinism.
    Reference:
        https://github.com/rlworkgroup/garage/blob/master/src/garage/examples/torch/trpo_pendulum.py
    """
    from garage import wrap_experiment
    from garage.experiment import Snapshotter
    from garage.experiment.deterministic import set_seed
    from garage.sampler import LocalSampler
    from garage.torch.algos import TRPO
    from garage.torch.policies import GaussianMLPPolicy
    from garage.torch.value_functions import GaussianMLPValueFunction
    from garage.trainer import Trainer

    @wrap_experiment(log_dir=log_dir, snapshot_mode='all')  # snapshot_mode: 'all', 'last'
    def train_wrapper(ctxt=None, seed=1):
        # set seed
        set_seed(seed)
        # make env
        env = init_env
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
        run_episodes(env=env, policy=trainer._algo.policy)

    def load_wrapper():
        # Load the env and policy from snap-shot
        snapshotter = Snapshotter()
        data = snapshotter.load(log_dir, itr='last')  # itr: iteration to load, an integer, 'last' or 'first'
        env = data['env']
        policy = data['algo'].policy
        run_episodes(env=env, policy=policy)

    if train is True:
        train_wrapper()
    else:
        load_wrapper()


def train_garage_tf(train: bool, log_dir: str, init_env):
    """ Train TRPO with Swimmer-v3 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce determinism.
        batch_size (int): Number of timesteps to use in each training step.
    Reference:
        https://github.com/rlworkgroup/garage/blob/master/src/garage/examples/tf/trpo_swimmer.py
    """
    from garage import wrap_experiment
    from garage.experiment import Snapshotter
    from garage.experiment.deterministic import set_seed
    from garage.np.baselines import LinearFeatureBaseline
    from garage.sampler import RaySampler
    from garage.tf.algos import TRPO
    from garage.tf.policies import GaussianMLPPolicy
    from garage.trainer import TFTrainer
    import tensorflow as tf

    @wrap_experiment(log_dir=log_dir, snapshot_mode='all')
    def train_wrapper(ctxt=None, seed=1, batch_size=1024):
        set_seed(seed)
        with TFTrainer(ctxt) as trainer:
            env = init_env
            policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            sampler = RaySampler(
                agents=policy, envs=env, max_episode_length=env.spec.max_episode_length, is_tf_worker=True
            )
            algo = TRPO(
                env_spec=env.spec, policy=policy, baseline=baseline, sampler=sampler, discount=0.99, max_kl_step=0.01
            )
            trainer.setup(algo, env)
            trainer.train(n_epochs=200, batch_size=batch_size)
            run_episodes(env=env, policy=trainer._algo.policy)

    def load_wrapper():
        snapshotter = Snapshotter()
        with tf.compat.v1.Session():
            data = snapshotter.load(log_dir, itr='last')
            env = data['env']
            policy = data['algo'].policy
            run_episodes(env=env, policy=policy)

    if train is True:
        train_wrapper()
    else:
        load_wrapper()


def run_episodes(env, policy):
    """ run 100 consecutive episodes and get avg reward """
    total_reward_list = []
    for e in range(100):
        observation, _ = env.reset()
        policy.reset()
        total_reward = 0.
        step = 0
        while True:  # gym swimmer: 1000 steps
            # env.render()
            action, _ = policy.get_action(observation)
            env_step = env.step(action)
            observation = env_step.observation
            total_reward += env_step.reward
            step += 1
            if env_step.last is True:
                break
        print('Episode {} finished after {} steps, reward: {}'.format(e, step, total_reward))
        total_reward_list.append(total_reward)
    env.close()
    print('{} episodes mean reward: {}'.format(len(total_reward_list), np.mean(total_reward_list)))


def init_gym_env():
    """ init gym env """
    from garage.envs import GymEnv
    return GymEnv('Swimmer-v3')


def test_garage(framework: str, train: bool, log_dir: str, init_env):
    """ Train/load and test RL algos from garage
    gym swimmer-v3 metrics for solved:
        avg reward over 100 consecutive episodes >= 360.0, maximum 1000 steps for each episode
    Benchmarking Deep Reinforcement Learning for Continuous Control (https://arxiv.org/abs/1604.06778) benchmarks:
        random: -1.7 +- 0.1
        TNPG: 96.0 +- 0.2
        TRPO: 96.0 +- 0.2
    TRPO: Trust Region Policy Optimization, policy gradient method
        output vars: mu, r both have the same dimension as a (action)
        output: [mu, r] = NeuralNet(s;{W_i, b_i}), neural net maps s (state) to mu (mean) and r (log std dev) with W_i and b_i (network params)
        sample action: a ~ N(mean=mu, stddev=exp(r)), normal distribution
    100 episodes mean reward: TRPO
        torch:
            - epochs 100, batch size 1024: 69.149 (video: moves slowly, but doesn't resemble swimming)
            - epochs 200, batch size 1024: 41.055
        tf:
            - epochs 40, batch size 4000: 32.620
            - epochs 200, batch size 1024: 40.810
    """
    if framework == 'torch':
        train_garage_torch(train=train, log_dir=log_dir, init_env=init_env)
    elif framework == 'tf':
        train_garage_tf(train=train, log_dir=log_dir, init_env=init_env)
    else:
        raise AssertionError


def run_episode_gym(env, policy, video_path):
    """ run 1 episode and record video with Gym VideoRecorder
    - this works with gym, but doesn't work with garage
    - video_path is a folder
    """
    env = gym.wrappers.Monitor(env, directory=video_path, force=True)

    observation, _ = env.reset()
    policy.reset()
    total_reward = 0.
    step = 0
    while True:
        # env.render()
        action, _ = policy.get_action(observation)
        env_step = env.step(action)
        observation = env_step.observation
        total_reward += env_step.reward
        step += 1
        if env_step.last is True:
            break
    print('Episode finished after {} steps, reward: {}'.format(step, total_reward))
    env.close()


def grab_frame_garage(env):
    """ get a frame with cv2 from garage's version of dm_control env """
    rgb_arr = env.render(mode='rgb_array')
    return cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB)


def run_episode_cv2(env, policy, video_path):
    """ run 1 episode and record video with cv2
    - video_path is a mp4 file that ends with '.mp4'
    """
    # Setup video writer
    frame = grab_frame_garage(env)
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
    video.write(frame)

    # run episodes
    observation, _ = env.reset()
    policy.reset()
    total_reward = 0.
    step = 0
    while True:
        # env.render()
        action, _ = policy.get_action(observation)
        env_step = env.step(action)
        video.write(grab_frame_garage(env))
        observation = env_step.observation
        total_reward += env_step.reward
        step += 1
        if env_step.last is True:
            break
    print('Episode finished after {} steps, reward: {}'.format(step, total_reward))
    env.close()
    video.release()


def record_garage(framework: str, log_dir: str, video_path: str, run_episode):
    """ Load RL algos from garage, run 1 episode and record video """
    from garage.experiment import Snapshotter

    def record_garage_torch():
        # Load the env and policy from snap-shot
        snapshotter = Snapshotter()
        data = snapshotter.load(log_dir, itr='last')  # itr: iteration to load, an integer, 'last' or 'first'
        env = data['env']
        policy = data['algo'].policy
        run_episode(env=env, policy=policy, video_path=video_path)

    def record_garage_tf():
        import tensorflow as tf

        # run episodes
        snapshotter = Snapshotter()
        with tf.compat.v1.Session():
            data = snapshotter.load(log_dir, itr='last')
            env = data['env']
            policy = data['algo'].policy
            run_episode(env=env, policy=policy, video_path=video_path)

    if framework == 'torch':
        record_garage_torch()
    elif framework == 'tf':
        record_garage_tf()
    else:
        raise AssertionError


if __name__ == '__main__':
    """ use register() to add a new environment, starts with -v0 """
    # print(gym.envs.registry.all())

    # take random actions and record video
    # test_random()

    # run RL algos from stable_baselines3
    # test_stable_baselines3(train=True)
    # test_stable_baselines3(train=False)

    # run RL algos from garage with torch
    # test_garage(framework='torch', train=True, log_dir='log/swimmer_gym_v3_trpo_torch', init_env=init_gym_env())
    # test_garage(framework='torch', train=False, log_dir='log/swimmer_gym_v3_trpo_torch', init_env=None)

    # run RL algos from garage with tf
    # test_garage(framework='tf', train=True, log_dir='log/swimmer_gym_v3_trpo_tf', init_env=init_gym_env())
    # test_garage(framework='tf', train=False, log_dir='log/swimmer_gym_v3_trpo_tf', init_env=None)

    # load garage torch/tf checkpoint, and record video
    record_garage(framework='torch', log_dir='log/swimmer_gym_v3_trpo_torch', video_path='video/swimmer_gym_v3_trpo_torch.mp4', run_episode=run_episode_cv2)
    # record_garage(framework='tf', log_dir='log/swimmer_gym_v3_trpo_tf', video_path='video/swimmer_gym_v3_trpo_tf.mp4', run_episode=run_episode_cv2)
