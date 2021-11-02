""" swimmer: closed-loop (feedback) control of forward locomotion
generate forward simulation dataset
"""

from virtual_nematode.envs.swimmer import make_swimmer
from virtual_nematode.models.forward import Forward
from virtual_nematode.data.simulation import generate_dataset


def model_kwargs_func(observation, **kwargs):
    return {'q': observation[1:12], 'q_vel': observation[15:]}


def x_func(stimuli, observation, **kwargs):
    return [stimuli] + observation[1:12].tolist() + observation[15:].tolist()


if __name__ == '__main__':
    """
    data_size: the total amount of sequences is data_size * (max_episode_steps / seq_len)
    reset_noise_scale: noise ~ U[-scale, scale]
        sampled noise is added to initial q and q_vel (unit: radian)
        joint range [-100, 100] degrees -> [-1.745, 1.745] rad
    max_episode_steps: the same amount of time for adapting random init pose to sine pose
        how long it takes depends on reset_noise_scale
    x: torch.Tensor, (max_episode_steps, stimuli_size + q_size + q_vel_size = input_size)
        external signal (first joint's target angle), proprioceptive observations (joint angles and angular velocity)
    y: torch.Tensor, (max_episode_steps, action_size)
        actions
    """
    input_size = 23  # 1 + 11 + 11
    data_size = 9000
    seed = 42
    max_episode_steps = 128
    reset_noise_scale = 1.745
    mode = 'sine_wave'  # or 'square_wave'
    save_name = 'data.pt'
    env = make_swimmer(max_episode_steps=max_episode_steps, reset_noise_scale=reset_noise_scale)
    model = Forward(dt=env.dt, seed=seed)
    generate_dataset(env, model, input_size, model_kwargs_func, x_func, data_size, seed, max_episode_steps, mode, save_name)
