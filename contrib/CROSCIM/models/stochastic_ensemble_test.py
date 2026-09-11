"""Test-time N-member ensembling for CROSCIM's stochastic solvers (CM/FM).

``StochasticEnsembleTestMixin`` overrides ``test_step``/``_finalize_res`` to
run ``self.n_test_members`` (default 1) independent stochastic samples per
batch, inside a single Lightning test pass — the dataloader/checkpoint are
loaded once, not once per member — writing one NetCDF per member (via
``aggregate_batches``'s ``member=`` kwarg). When ``n_test_members <= 1`` both
overrides delegate straight to ``super()``, so nothing changes for existing
configs or for the shared deterministic-solver code path in ``models.py`` —
this mixin only does anything once a subclass explicitly sets
``n_test_members > 1``.

For multi-resolution xps, members are PAIRED across resolutions: the finer
resolution's member ``m`` uses the coarser resolution's member ``m`` (not an
arbitrary/shared coarse field), so each member is a self-consistent
multi-res sample. This is why ``self.aggregate_results[res_key]`` becomes a
list of ``n_test_members`` datasets instead of a single one once this mixin
is active for that resolution.

Usage: have the stochastic Lightning module inherit from this mixin (in
addition to its normal base), and set ``self.n_test_members`` from config in
``__init__``.
"""
from __future__ import annotations

import glob
import itertools
import os

import numpy as np
import torch
import xarray as xr


def write_spread_netcdf(lit_mod, res, n_members, write_netcdf=True):
    """After ``_finalize_res`` has written one NetCDF per member (via
    ``aggregate_batches(..., member=m)``), read them back and write one more
    NetCDF per test window where every ``pred_<var>`` is replaced by
    ``spread_<var>`` = std across the ``n_members`` values. ``tgt_<var>``/obs
    fields are taken from member 0 unchanged (deterministic, don't vary by
    member). No-op when there is nothing to merge (n_members <= 1,
    write_netcdf=False, or no logger)."""
    if not (write_netcdf and lit_mod.logger and n_members > 1):
        return
    log_dir = lit_mod.logger.log_dir
    member0_files = sorted(glob.glob(os.path.join(log_dir, f"test_data_*_patch_x{res}_member0.nc")))
    for f0 in member0_files:
        out_path = f0[: -len("_member0.nc")] + ".nc"
        member_paths = [f0[: -len("_member0.nc")] + f"_member{m}.nc" for m in range(n_members)]
        member_ds = [xr.open_dataset(p) for p in member_paths]

        pred_vars = [v for v in member_ds[0].data_vars if v.startswith("pred_")]
        merged = member_ds[0].copy(deep=True)
        for var in pred_vars:
            spread = xr.concat([ds[var] for ds in member_ds], dim="member").std(dim="member")
            del merged[var]
            merged[f"spread_{var[len('pred_'):]}"] = spread

        merged.to_netcdf(out_path)
        print(f"[ensemble] wrote {out_path} (spread over {n_members} members)")
        for ds in member_ds:
            ds.close()


class StochasticEnsembleTestMixin:

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        n_members = getattr(self, "n_test_members", 1)
        if n_members <= 1 or self._is_simple_datamodule():
            return super().test_step(batch, batch_idx, dataloader_idx)

        if dataloader_idx is None:
            dataloader_idx = 0
        res = self.multires[dataloader_idx]
        res_key = f"patch_x{res}"
        last = self.len_daw[res]

        print(f"Dataloader_{dataloader_idx}, Batch_{batch_idx}, res_{res} (n_members={n_members})")
        if (dataloader_idx == 0) and (batch_idx == 0):
            self.test_data = {}
            self.test_times = {}
            self.test_coords = {}
            self.aggregate_results = {}

        if batch_idx == 0:
            self.test_data[res_key] = [[] for _ in range(n_members)]
            self.test_times[res_key] = [[] for _ in range(n_members)]
            self.test_coords[res_key] = [[] for _ in range(n_members)]

        orig_batch = self.modify_batch(batch, res)
        device = (orig_batch.tgt_sic.device if hasattr(orig_batch, 'tgt_sic')
                  else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        for m in range(n_members):
            # Read by FM/CM's forward() to append to _bound_inputs only once
            # per batch (on member 0) instead of once per member — the
            # observations it stores don't vary across members.
            self._ensemble_member_idx = m
            batch_m = orig_batch

            if dataloader_idx > 0:
                coarser_res = self.multires[dataloader_idx - 1]
                xc_target = torch.squeeze(batch_m.xc, dim=1)
                yc_target = torch.squeeze(batch_m.yc, dim=1)
                # Paired member: this resolution's member m builds on the
                # coarser resolution's own member m (aggregate_results[...]
                # is a list of n_members datasets once this mixin is active).
                coarse = self.aggregate_results[f"patch_x{coarser_res}"][m]
                coarse = {
                    k: v.isel(time=np.arange(self.len_daw[coarser_res] - last,
                                              self.len_daw[coarser_res]))
                    for k, v in coarse.items()
                }
                coarse = self.convert_xr_to_batch(coarse, batch_m, res=res)
                coarse = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in coarse.items() if v is not None
                }
                xc_coarse = torch.squeeze(coarse["xc"], dim=1)
                yc_coarse = torch.squeeze(coarse["yc"], dim=1)
                itrp_coarse = self.interpolate_torch(coarse, xc_coarse, yc_coarse,
                                                      xc_target, yc_target)
                batch_m = self.update_batch_as_anomaly(
                    batch_m, {k: v for k, v in itrp_coarse.items() if k.startswith('pred_')}
                )
                tgt_vars_for_res = self._get_target_vars_for_resolution(res)
                orig_tgt = {var: getattr(batch_m, var).clone() for var in tgt_vars_for_res}
                if self.normalize_anomaly:
                    batch_m, anom_scale = self.normalize_anomaly_batch(batch_m, res=res)
                else:
                    anom_scale = {}
                scale_channel = None
                if self.condition_on_scale:
                    scale_channel = self.compute_scale_channel(
                        {k: v for k, v in itrp_coarse.items() if k.startswith('pred_')}
                    )
            else:
                anom_scale = {}
                orig_tgt = None
                scale_channel = None
                itrp_coarse = None

            sbatch = self.format_batch_for_solver(
                batch_m, include_masks=self.include_masks, res=res, scale_channel=scale_channel
            )

            out = self(batch=sbatch, res=res)
            out = self.split_tensor_to_dict(out, res=res)

            if anom_scale:
                out = self.denormalize_anomaly_predictions(out, anom_scale)

            tgt_vars = self._get_target_vars_for_resolution(res)

            if dataloader_idx > 0:
                out = {k: out[k] + itrp_coarse[k] for k in out}

            batch_dict = batch_m._asdict()
            models_mask_var = next(
                (k for k in batch_dict if k.startswith("models_")
                 and isinstance(batch_dict[k], torch.Tensor)
                 and batch_dict[k].numel() > 0),
                None
            )
            if models_mask_var is not None:
                domain_invalid = ~batch_dict[models_mask_var].isfinite().any(dim=1, keepdim=True)
            else:
                domain_invalid = (batch_m.land_mask == 1.)

            for var in out:
                out[var] = torch.where(
                    domain_invalid,
                    torch.tensor(float('nan'), device=out[var].device, dtype=out[var].dtype),
                    out[var],
                )

            out_norm, tgt_norm = {}, {}
            for var in tgt_vars:
                if '_' in var:
                    var_suffix = var.split('_', 1)[1]
                    pred_var_name = f'pred_{var_suffix}'
                else:
                    pred_var_name = f'pred_{var}'
                pred = out[pred_var_name]
                out_norm[pred_var_name] = pred
                if orig_tgt is not None:
                    tgt_norm[var] = orig_tgt[var]
                else:
                    tgt_norm[var] = getattr(batch_m, var)
                if dataloader_idx > 0:
                    tgt_norm[var] = tgt_norm[var] + itrp_coarse[pred_var_name]

            out_norm = self._apply_constraints(out_norm, res)

            obs_norm = {field: getattr(batch_m, field)
                        for field in self._get_obs_var_names(dataloader_idx)}

            combined = list(out_norm.values()) + list(tgt_norm.values()) + list(obs_norm.values())
            stacked = torch.stack(combined, dim=1)

            self.test_data[res_key][m].append(stacked)
            self.test_times[res_key][m].append(torch.squeeze(batch_m.time, dim=1))

            batch_size = stacked.shape[0]
            patch_coords_batch = []
            for b in range(batch_size):
                yc_b = batch_m.yc[b].squeeze().detach().cpu().numpy()
                xc_b = batch_m.xc[b].squeeze().detach().cpu().numpy()
                patch_coords_batch.append((yc_b, xc_b))
            self.test_coords[res_key][m].append(patch_coords_batch)

        if self.is_last_batch(batch_idx, dataloader_idx):
            idx_rec = np.arange(orig_batch.time.shape[-1])
            self._finalize_res(dataloader_idx, idx_rec, write_netcdf=True)

    def _finalize_res(self, dataloader_idx, idx_rec, write_netcdf=True):
        n_members = getattr(self, "n_test_members", 1)
        if n_members <= 1:
            return super()._finalize_res(dataloader_idx, idx_rec, write_netcdf=write_netcdf)

        res = self.multires[dataloader_idx]
        res_key = f"patch_x{res}"

        results = []
        for m in range(n_members):
            data = list(itertools.chain(*self.test_data[res_key][m]))
            times = list(itertools.chain(*self.test_times[res_key][m]))
            coords = list(itertools.chain(*self.test_coords[res_key][m]))

            if self.trainer.world_size > 1:
                import torch.distributed as dist
                gathered = [None] * self.trainer.world_size
                dist.all_gather_object(
                    gathered,
                    {'data': data, 'times': [t.cpu() for t in times], 'coords': coords},
                )
                data = [item for g in gathered for item in g['data']]
                times = [t for g in gathered for t in g['times']]
                coords = [c for g in gathered for c in g['coords']]

                if self.trainer.is_global_zero:
                    result = self.aggregate_batches(
                        idx_rec, data, times, dataloader_idx,
                        metrics=False, write_netcdf=write_netcdf,
                        patch_coords=coords, member=m,
                    )
                    print(result)
                else:
                    result = None
                container = [result]
                dist.broadcast_object_list(container, src=0)
                results.append(container[0])
            else:
                result = self.aggregate_batches(
                    idx_rec, data, times, dataloader_idx,
                    metrics=False, write_netcdf=write_netcdf,
                    patch_coords=coords, member=m,
                )
                print(result)
                results.append(result)

        if self.trainer.world_size <= 1 or self.trainer.is_global_zero:
            write_spread_netcdf(self, res, n_members, write_netcdf=write_netcdf)

        self.aggregate_results[res_key] = results
