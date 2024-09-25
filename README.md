### Neuromod RNN
This repository contains a JAX implementation of neuromodRNN, which combines e-prop[1] update with diffusion of neurotransmitters (which carry learning signals). The current repostory is still under development and no tutorial 
notebooks are available.

The jax e-prop implementation of this repository is inspired in the implementation of original e-prop paper authors: https://github.com/IGITUGraz/eligibility_propagation


[[1] A solution to the learning dilemma for recurrent networks of spiking neurons G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass.](https://www.nature.com/articles/s41467-020-17236-y#code-availability)

### Running
The various implemented methods can be run for different tasks through main.py by setting appropriate flags to override the default configurations. More details will be added in the future.

### Installation
The detailed list of required packages is provided in the environment.yml file. You can reproduce the environment using Conda with the following command:

```
conda env create -f environment.yml -n env_name
```
**Note:** The JAX version listed in environment.yml is CPU-only. To run on a GPU, it is necessary to upgrade to the GPU version (see: [JAX installation](https://jax.readthedocs.io/en/latest/installation.html)]. To run
`main.py` file within an enivronment with JAX for GPU installed on a node with only CPU available it might be necessary to set the global variable `JAX_PLATFORMS=cpu`.
