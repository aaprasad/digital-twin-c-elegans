# virtual-nematode

Virtual C. elegans simulations with PyTorch neural networks and MuJoCo environments

## env for C. elegans

1. swimmer
    * `swimmer_gym_v3`: OpenAI Gym's Swimmer-v3
    * `swimmer_gym_v3_v0`: OpenAI Gym's Swimmer-v3
    * `swimmer_dm`: dm_control's swimmer
2. swimmer: n_links and joint_range
    * `swimmer_gym_v3_v1`: based on OpenAI Gym's Swimmer-v3
    * `swimmer_dm_v0`: dm_control's swimmer
3. swimmer: n_links, joint_range and body_len
    * `swimmer_gym_v3_v2`
4. muscle swimmer
    * `muscle_swimmer_gym_v0`: based on OpenAI Gym's Swimmer-v3
5. muscle swimmer: n_links and joint_range
    * `muscle_swimmer_gym_v1`
6. muscle swimmer: n_links, joint_range, body_len and muscle_len
    * `muscle_swimmer_gym_v2`
7. nematode: joint_range, body_len, muscle_len and arrangement
    * `nematode_gym_v0`

## env wrapper for specific tasks

1. chemotaxis: a cylinder as chemical source, no boundaries
    * `chemotaxis`
2. forage: cylinders as food, box geoms as square boundary
    * `forage`
3. maze: box geoms as walls and square boundary
    * `maze`

## gym wrapper

1. distribution: create a chemical distribution and provide concentration and gradient
    * `distribution`
2. recorder: record a video
    * `recorder`

## mathematical models of locomotion

1. computational model: chemotaxis with behaviors including forward, pirouette, weathervane and random walk
    * `computational_model`
2. forward: forward sinusoidal movement
    * `forward`

## networks

### ncp

1. swimmer chemotaxis
    * chemotaxis simulation
        ```
        python swimmer_chemotaxis.py
        ```
    * generate chemotaxis dataset
        ```
        python swimmer_chemotaxis_data.py
        ```
    * preprocess chemotaxis dataset for offline training
        ```
        python swimmer_chemotaxis_ncp_data.py
        ```
    * offline training and testing
        ```
        pyhton swimmer_chemotaxis_ncp.py
        ```
    * check TensorBoard log
        ```
        tensorboard --logdir=runs --host=10.176.50.34 --port=6006
        ```
    * online testing
        ```
        python swimmer_chemotaxis_ncp_online.py
        ```
2. swimmer forward
    ```
    python swimmer_forward.py
    python swimmer_forward_data.py
    python swimmer_forward_ncp_data.py
    python swimmer_forward_ncp.py
    python swimmer_forward_ncp_online.py
    ```
