""" muscle worm: closed-loop (feedback) control of forward locomotion """

from gym_worm.wrappers.muscle_action import joint_to_muscle_action, JointAction
import os
from sim import action_func
import torch
from virtual_nematode.envs.muscle_worm import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.data.simulation import generate_dataset


def x_func(observation, **kwargs):
    """ x: input_size = tendon_length_size + tendon_velocity_size
    external signal: first joint's target angle
    proprioceptive observations: tendon lengths and velocity
    """
    # return [stimuli] + observation[52:148].tolist() + observation[148:244].tolist()
    return observation[52:148].tolist() + observation[148:244].tolist()


def y_func(action, **kwargs):
    """ y: action_size = 96
    ctrl signal: joint action -> muscle action
    """
    action = joint_to_muscle_action(action)
    return action.tolist()


if __name__ == '__main__':
    input_size = 192
    data_size = 6000
    seed = 7
    max_episode_steps = 192
    reset_noise_scale = 0.7
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale)
    action_size = env.action_space.shape[0]  # original muscle action size
    env = JointAction(env)  # ctrl: joint -> muscle
    model = Forward(dt=env.dt, seed=None, n=25, q_max=20., a_max=1., psi=0.1, freq=1.5)
    dataset = generate_dataset(env, model, action_func, x_func, y_func, input_size, action_size, data_size, max_episode_steps, seed)
    os.makedirs('data', exist_ok=True)
    torch.save(dataset, 'data/data.pt')
