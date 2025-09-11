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
from contrib.ASIP_OSISAF.load_data import *
import datetime
import pyresample
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import os
from torch.utils.data.sampler import Sampler

TrainingItem = namedtuple(
    'TrainingItem', ['asip', 'osisaf', 
                     'lat', 'lon', 'land_mask',
                     't2m','istl1','sst','skt']
)

TrainingItem_4da = namedtuple(
    'TrainingItem_4da', ['asip', 'osisaf', 'input',
                     'lat', 'lon', 'land_mask',
                     't2m','istl1','sst','skt']
)

ExtendedTrainingItem_4da = namedtuple(
    'ExtendedTrainingItem_4da', ['asip', 'osisaf',
                     'tgt', 'coarse', 'input',
                     'lat', 'lon', 'latv', 'lonv', 
                     'land_mask',
                     't2m','istl1','sst','skt']
)

class XrDataset(torch.utils.data.Dataset):

    'Characterizes a dataset for PyTorch'
    def __init__(self, path, split, da, build_batch=None):
        'Initialization'
        self.db = xr.open_dataset(path).isel(record=split)
        self.da = da
        self.build_batch = build_batch

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.db.asip)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        
        item = self.db.isel(record=item,sample=0)
        if self.da:
            tr_item = TrainingItem_4da
        else:
            tr_item = TrainingItem
        item = item[[*tr_item._fields]].to_array()
        """
        if self.da:
            # Assuming da is your xarray.DataArray with a 'variable' coordinate
            replacements = {'asip': 'tgt', 'osisaf': 'coarse'}
            # Vectorized replacement function
            def replace_vars(var):
                return replacements.get(var, var)  # Keep var unchanged if not in dict
            item = item.assign_coords(variable=(item.coords['variable'].dims,
                                                np.vectorize(replace_vars)(item.coords['variable'].values)
                                               )
                                     )
        """
        item = item.data.astype(np.float32)
        if self.da:
            item = self.build_batch(item)
            item_dict = item._asdict()
            item_dict["tgt"] = item.asip
            item_dict["coarse"] = item.osisaf
            item_dict["lonv"] = item.lon
            item_dict["latv"] = item.lat
            item = ExtendedTrainingItem_4da(**item_dict)

        return item

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, asip_paths,
                 split_train,
                 split_val,
                 split_test,
                 norm_stats,
                 norm_stats_covs,
                 da=False,
                 **kwargs):
        
        super().__init__()
        self.asip_paths = asip_paths
        self.split_train = split_train
        self.split_val = split_val
        self.split_test = split_test
        self.da = da
        self._norm_stats = norm_stats
        self._norm_stats_covs = norm_stats_covs

    def norm_stats(self):
        return self._norm_stats

    def norm_stats_covs(self):
        return self._norm_stats_covs

    def build_batch(self):
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem_4da._make,
            lambda item: item._replace(input=item.input),
            lambda item: item._replace(osisaf=item.osisaf),
            lambda item: item._replace(asip=item.asip),
            lambda item: item._replace(land_mask=item.land_mask),
            lambda item: item._replace(lat=item.lat),
            lambda item: item._replace(lon=item.lon),
            lambda item: item._replace(t2m=item.t2m),
            lambda item: item._replace(istl1=item.istl1),
            lambda item: item._replace(sst=item.sst),
            lambda item: item._replace(skt=item.skt)
            ])

    def setup(self, stage='test'):
        build_batch = self.build_batch()
        self.train_ds = XrDataset(self.asip_paths, self.split_train, self.da,
                                  build_batch)
        self.val_ds = XrDataset(self.asip_paths, self.split_val, self.da,
                                build_batch)
        self.test_ds = XrDataset(self.asip_paths, self.split_test, self.da,
                                 build_batch)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, batch_size=2,
                                           num_workers=5, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, batch_size=2,
                                           num_workers=5, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, batch_size=2,
                                           num_workers=5, persistent_workers=True)
