# virtual-nematode
Virtual C. elegans simulations with PyTorch neural networks and MuJoCo environments

## prerequisites
* install `gym-worm`
* install the project in editable mode from local project path
    ```
    pip install .
    ```

## mathematical models of locomotion
* `computational_model`: chemotaxis with behaviors including forward, pirouette, weathervane and random walk
* `forward`: forward sinusoidal movement

## data
* `concat`: concatenate torch.utils.data.TensorDataset
* `subset.RandomSubset`: sample a random subset
* `subset.FilterSubset`: sample a subset with higher chemotaxis index
* `split`: split a time sequence
* `chemotaxis`: create a dataset of chemotaxis locomotion
* `simulation.SimulationSample`: generate one simulation sample
* `simulation.SimulationDataset`: do parallel simulations by multiprocessing and collect the samples

## networks
### ncp
* swimmer_chemotaxis
    ```
    python swimmer_chemotaxis.py
    python swimmer_chemotaxis_data.py
    python swimmer_chemotaxis_ncp_data.py
    pyhton swimmer_chemotaxis_ncp.py
    python swimmer_chemotaxis_ncp_online.py
    ```
* swimmer_forward
    ```
    python swimmer_forward.py
    python swimmer_forward_data.py
    python swimmer_forward_ncp_data.py
    python swimmer_forward_ncp.py
    python swimmer_forward_ncp_online.py
    ```

## TensorBoard
* check TensorBoard log
    ```
    tensorboard --logdir=runs --host=10.176.50.34 --port=6006
    ```
