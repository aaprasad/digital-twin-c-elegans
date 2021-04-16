# virtual-nematode

## env for C. elegans

1. existing env
    * OpenAI Gym's `Swimmer-v3`: `swimmer_gym_v3`
    * dm_control's `swimmer`: `swimmer_dm`
2. existing env with any specified `n_links`
    * OpenAI Gym's `Swimmer-v3` -> `swimmer_gym_v3_v1`
    * dm_control's `swimmer`: `swimmer_dm_v0`
3. existing env with any specified `n_links` and `body_len`
    * `swimmer_gym_v3_v1` -> `swimmer_gym_v3_v2`
4. muscle env (*)
    * OpenAI Gym's `Swimmer-v3` -> `muscle_swimmer_gym_v0`
5. muscle env with any specified `n_links` (*)
    * `muscle_swimmer_gym_v0` -> `muscle_swimmer_gym_v1`
6. muscle env with any specified `n_links` and `body_len` (*)
    * `muscle_swimmer_gym_v1` -> `muscle_swimmer_gym_v2`
7. muscle nematode env (*)
    * `muscle_swimmer_gym_v2` -> `nematode_gym_v0`
