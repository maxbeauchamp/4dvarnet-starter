
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import xarray as xr
import ocean4dvarnet.data
import xarray as xr
import os


def rmse_based_scores(ds):

    da_rec = ds["pred_sic"]
    da_ref = ds["tgt_sic"]


    # RMSE globale
    rmse = np.sqrt(((da_rec - da_ref) ** 2).mean(skipna=True))

    return (
        np.round(rmse.values, 5).item()
    )