""" OpenAI Gym's Swimmer-v3
SwimmerEnv works the same as Swimmer-v3
    - different default xml_file
    - https://github.com/openai/gym/blob/master/gym/envs/mujoco/swimmer_v3.py
"""

import os
from gym.envs.mujoco import mujoco_env, swimmer_v3
from gym import utils


class SwimmerEnv(swimmer_v3.SwimmerEnv):
    def __init__(self,
                 xml_file=os.path.join(os.path.dirname(__file__), 'assets', 'swimmer.xml'),
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-4,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        """ override swimmer_v3.SwimmerEnv """
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 4)
