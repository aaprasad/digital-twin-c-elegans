# virtual-nematode
Virtual C. elegans simulations with PyTorch neural networks and MuJoCo environments

## prerequisites
* install `gym-worm`
* install the project in editable mode from local project path
    ```
    pip install -e .
    ```
* change local packages to be built in-place without first copying to a temporary directory
    ```
    pip install -e . --use-feature=in-tree-build
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
* run simulation
    ```
    python sim.py
    ```
* generate simulation data
    ```
    python data.py
    ```
* preprocess data
    ```
    python ncp_data.py
    ```
* train and test network
    ```
    python ncp.py
    ```
* online test
    ```
    python ncp_online.py
    ```

## TensorBoard
* check TensorBoard log
    ```
    tensorboard --logdir=runs --host=10.176.50.34 --port=6006
    ```
