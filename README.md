# virtual-nematode
Virtual C. elegans simulations with PyTorch neural networks and MuJoCo environments

## mathematical models of locomotion
1. computational model: chemotaxis with behaviors including forward, pirouette, weathervane and random walk
    * `computational_model`
2. forward: forward sinusoidal movement
    * `forward`

## data
1. concat dataset: concatenate torch.utils.data.TensorDataset
    * `concat`
2. filter subset: sample a subset
    * `subset`: `RandomSubset` sample a random subset, `FilterSubset` sample a subset with higher chemotaxis index
3. split dataset: split a time sequence
    * `split`
4. chemotaxis dataset: create a dataset of chemotaxis locomotion
    * `chemotaxis`
5. simulation dataset: do parallel simulations by multiprocessing
    * `simulation`: `SimulationSample` generates one sample, `SimulationDataset` collects the samples

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
