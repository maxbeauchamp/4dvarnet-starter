# 4DVarNet

## Prerequisite
- git
- conda

## Install
### Install project dependencies
```
git clone https://github.com/CIA-Oceanix/4dvarnet-starter.git
cd 4dvarnet-starter
conda install -c conda-forge mamba
conda create -n 4dvarnet-starter
conda activate 4dvarnet-starter
mamba env update -f environment.yaml
```

```

## Run
The model uses hydra see [#useful-links]
```
python main.py xp=base 
```

## Stochastic extensions

We build here a stochastic extension of the 4DVarNet framework by replacing the regularization term of the variational cost with a stochastic component inherited either from analog or SPDE-based framework to provide a generative feature associated to 4DVarNet.

### Analog-based UQ
![Analog based generative modeling](figs/En4DVarNet-analog.png)

### SPDE-based generative modeling

![Advection-diffusion based generative modeling](figs/En4DVarNet-gen.png)


## Useful links:
- [Hydra documentation](https://hydra.cc/docs/intro/)
- [Pytorch lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/index.html#get-started)
- 4DVarNet papers:
  - Beauchamp, M., R. Fablet, S. Benaichouche, P. Tandeo, N. Desassis, and B. Chapron, 2025: Neural variational Data Assimilation with Uncertainty Quantification using SPDE priors. Artif. Intell. Earth Syst., https://doi.org/10.1175/AIES-D-24-0060.1, in press. 
