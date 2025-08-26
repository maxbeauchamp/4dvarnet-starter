# 4DVarNet

## Prerequisite
- git
- conda

## Install
### Install project dependencies
```
git clone https://github.com/maxbeauchamp/4dvarnet-sst
cd 4dvarnet-sst
conda install -c conda-forge mamba
conda create -n 4dvarnet-sst
conda activate 4dvarnet-sst
mamba env update -f environment.yaml
```

## Run
The model uses hydra see [#useful-links]. You can redefined experiments after adpating the configuration files.
```
python main.py xp=anom/dmi_sst_all_baltic_wcoarse_wgeo.yaml
```

## Improvement compared to DMI-OI

![comparison of gradSST for DMI-OI vs 4DVarNet](figs/comparison_OI_4DVarNet_grad.png)


## Useful links:
- [Hydra documentation](https://hydra.cc/docs/intro/)
- [Pytorch lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/index.html#get-started)


