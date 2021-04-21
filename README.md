# virtual-nematode

## env for C. elegans

1. swimmer
    * OpenAI Gym's `Swimmer-v3`: `swimmer_gym_v3`
    * dm_control's `swimmer`: `swimmer_dm`
2. swimmer with specific `n_links` and `joint_range`
    * OpenAI Gym's `Swimmer-v3` -> `swimmer_gym_v3_v1`
    * dm_control's `swimmer`: `swimmer_dm_v0`
3. swimmer with specific `n_links`, `joint_range` and `body_len`
    * `swimmer_gym_v3_v1` -> `swimmer_gym_v3_v2`
4. muscle swimmer
    * OpenAI Gym's `Swimmer-v3` -> `muscle_swimmer_gym_v0`
5. muscle swimmer with specific `n_links` and `joint_range`
    * `muscle_swimmer_gym_v0` -> `muscle_swimmer_gym_v1`
6. muscle swimmer with specific `n_links`, `joint_range` and `body_len`
    * `muscle_swimmer_gym_v1` -> `muscle_swimmer_gym_v2`
7. nematode with specific `joint_range`, `body_len` and `arrangement`
    * `muscle_swimmer_gym_v2` -> `nematode_gym_v0`
