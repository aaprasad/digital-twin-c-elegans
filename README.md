# virtual-nematode
Virtual C. elegans simulations with PyTorch neural networks and MuJoCo environments

## prerequisites
* python==3.8
* pytorch==1.8.0
* tensorboard==2.4.1

## installation
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
* `tap`: tap-withdrawal composing of forward, backward and stochastic turning behaviors

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
#### how to use multiple GPUs using torch.nn.DataParallel
* DataParallel tutorial: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
* All manually defined parameters need to be registered, so that they can be transferred along with the module to any devices.
* torch.nn.ParameterDict can register dict of parameters, but it doesn't support DataParallel.
* In torch.nn.Module, register parameters one by one with `self.register_parameter(name, param)`, and access `param` directly with `self.name`.
* save DataParallel models by:
    ```
    torch.save(model.module.state_dict(), PATH)
    ```
#### usage
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
* online test the trained network
    ```
    python ncp_online.py
    ```

## TensorBoard
* check TensorBoard log
    ```
    tensorboard --logdir=runs --host=10.176.50.34 --port=6006
    ```
* local
    ```
    tensorboard --logdir=runs
    ```
