from random import sample
import contrib
from contrib.CROSCIM.load_data import *
from contrib.CROSCIM.data_simple import *
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

class XrDatasetMultiRes_simplify(torch.utils.data.Dataset):

    'Characterizes a dataset for PyTorch'
    def __init__(self, paths, split, multires, build_batch):

        'Initialization'
        self.multires = multires
        self.db = {}
        for res in self.multires:
            self.db[f"patch_x{res}"] = xr.open_dataset(paths[f"patch_x{res}"]).isel(record=split)
        self.build_batch = build_batch

    def __len__(self):
        'Denotes the total number of samples'
        res_min = self.multires[-1]
        return len(self.db[f"patch_x{res_min}"].asip_sic)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):

        out = {}
        for res in self.multires:
            item = self.db[f"patch_x{res}"].isel(record=idx,sample=0)
            item = item[[*TrainingItem._fields]]
            var_dict = {var: item[var].values for var in item.data_vars}
            var_dict["time"] = item.time.data
            var_dict["xc"] = item.xc.data
            var_dict["yc"] = item.yc.data
            item = self.build_batch(var_dict)
            out[f"patch_x{res}"] = item
        return out

class BaseDataModuleMultiRes_simplify(pl.LightningDataModule):
    def __init__(self, croscim_preproc_paths,
                 multires,
                 split_train,
                 split_val,
                 split_test,
                 norm_stats,
                 norm_stats_covs,
                 **kwargs):

        super().__init__()
        self.croscim_preproc_paths = croscim_preproc_paths
        self.multires = multires
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
        self.train_ds = XrDatasetMultiRes_simplify(self.croscim_preproc_paths, 
                                          self.split_train, self.multires, build_batch)
        self.val_ds = XrDatasetMultiRes_simplify(self.croscim_preproc_paths,
                                        self.split_val, self.multires, build_batch)
        self.test_ds = XrDatasetMultiRes_simplify(self.croscim_preproc_paths,
                                         self.split_test, self.multires, build_batch)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, batch_size=2,
                                           num_workers=5, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, batch_size=2,
                                           num_workers=5, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, batch_size=2,
                                           num_workers=5, persistent_workers=True)


