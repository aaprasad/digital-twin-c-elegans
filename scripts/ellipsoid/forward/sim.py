import numpy as np
from virtual_nematode.envs.ellipsoid import make_swimmer
from virtual_nematode.models.forward import ForwardZY
from virtual_nematode.simulation import simulate


def action_func(model, step, observation, **kwargs):
    q = observation[5:53][1::2]
    q_vel = observation[58:106][1::2]
    action = model.step(step, q, q_vel)
    return action


def step_func(observation, **kwargs):
    com = observation[106:108]  # 2D center of mass
    return com


def done_func(index, result, **kwargs):
    """
    result: list of com in 1 simulation
    return: displacement in this simulation
    """
    displacement = np.linalg.norm(np.array(result[-1]) - np.array(result[0]), ord=2)
    print('Trial {}: com displacement {:.2f}'.format(index + 1, displacement))
    return displacement


if __name__ == '__main__':
    max_episode_steps = 2500  # 0.04s/step, 100s in total
    # condim=1: no friction; condim=3: tangential and normal sliding friction
    env = make_swimmer(
        n_bodies=25, joint_range='-100 100', max_episode_steps=max_episode_steps, reset_noise_scale=0.,
        density=4000, viscosity=0.1, condim=3, friction='0.1 1'
    )
    # env = gym.wrappers.Monitor(env, directory='video/swimmer', force=True)
    print(env.action_space)
    print(env.observation_space)
    y_ctrl = np.zeros(24)
    # q_max=40, psi=1.54, freq=0.8
    model = ForwardZY(y_ctrl, dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.05, freq=0.8, motor=True)
    results = simulate(env, model, action_func, step_func, done_func, seed=None, trials=1, render=False)
    print('{} trials: com displacement mean {:.2f} / {} steps'.format(len(results), np.mean(results), max_episode_steps))
