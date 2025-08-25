import pytorch_lightning as pl
import numpy as np
import torch.utils.data
import torch
import xarray as xr
import itertools
import functools as ft
import tqdm
from collections import namedtuple
from torch.utils.data import  ConcatDataset
import multiprocessing
import gc
from random import sample
import contrib
from contrib.CROSCIM.load_data import *
import datetime
import pyresample
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import os
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F

def create_training_item(var_groups, covariates, tgt_vars):
    """
    Dynamically create a TrainingItem with variables from var_groups and covariates
    """
    fields = []
    # Variables satellites
    for group in var_groups:
        for var in var_groups[group]:
            fields.append(f"{group}_{var}")
    for group, variables in VAR_GROUPS.items():
        for var in variables:
            var_key = f"{group}_{var}"
            new_key = f"tgt_{var}"
            if (var_key in tgt_vars):
                fields.append(new_key)
    # Covariates
    fields.extend(covariates)
    # Coordonnées et masque
    fields.extend(['lat', 'lon', 'land_mask', "time", "yc", "xc"])
    # Final namedtuple
    return namedtuple("TrainingItem", fields)

TrainingItem = create_training_item(VAR_GROUPS, COVARIATES,
                                    tgt_vars=["asip_sic","cimr_SIT"])

class XrDataset_simplify(torch.utils.data.Dataset):

    'Characterizes a dataset for PyTorch'
    def __init__(self, path, split, build_batch):
        'Initialization'
        self.db = xr.open_dataset(path).isel(record=split)
        self.build_batch = build_batch

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.db.asip_sic)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):

        item = self.db.isel(record=idx,sample=0)
        item = item[[*TrainingItem._fields]]
        var_dict = {var: item[var].values for var in item.data_vars}
        var_dict["time"] = item.time.data
        var_dict["xc"] = item.xc.data
        var_dict["yc"] = item.yc.data
        item = self.build_batch(var_dict)

        return item

class BaseDataModule_simplify(pl.LightningDataModule):
    def __init__(self, croscim_preproc_path,
                 split_train,
                 split_val,
                 split_test,
                 norm_stats,
                 norm_stats_covs,
                 **kwargs):

        super().__init__()
        self.croscim_preproc_path = croscim_preproc_path
        self.split_train = split_train
        self.split_val = split_val
        self.split_test = split_test
        self._norm_stats = norm_stats
        self._norm_stats_covs = norm_stats_covs

    def norm_stats(self):
        return self._norm_stats

    def norm_stats_covs(self):
        return self._norm_stats_covs

    def build_batch(self, item_dict):
        # Extract only the fields defined in the TrainingItem
        fields = {k: v for k, v in item_dict.items() if k in TrainingItem._fields}
        return TrainingItem(**fields)

    def setup(self, stage='test'):
        build_batch = self.build_batch
        self.train_ds = XrDataset_simplify(self.croscim_preproc_path, self.split_train, build_batch)
        self.val_ds = XrDataset_simplify(self.croscim_preproc_path, self.split_val, build_batch)
        self.test_ds = XrDataset_simplify(self.croscim_preproc_path, self.split_test, build_batch)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, batch_size=2,
                                           num_workers=5, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, batch_size=2,
                                           num_workers=5, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, batch_size=2,
                                           num_workers=5, persistent_workers=True)




