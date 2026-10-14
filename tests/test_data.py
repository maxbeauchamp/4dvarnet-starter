"""Phase 2 end-to-end test of the cascade dataloader on small synthetic Zarr
stores (only the chunks around the test tile are written). Run from the repo
root, e.g.:

    pytest tests/test_data.py
"""
import os

import numpy as np
import pytest
import torch
import zarr

from croscim import data as D
from croscim import grid as g
from croscim import obs
from croscim.asip import asip_to_m

DAYS = list(range(19000 - D.W + 1, 19001))
IY0, IX0 = 0, 68  # top row of the domain: partly beyond the 5 km stores (y > 2475 km)


def _write_stores(root):
    rng = np.random.default_rng(0)
    nd = len(DAYS)
    n5 = 12800
    f = 64
    sl = (slice(IY0 * f, (IY0 + D.N_C) * f), slice(IX0 * f, (IX0 + D.N_C) * f))
    asip = os.path.join(root, D.STORES["asip"])
    for name, dtype, fill in (("sic", "u1", 255), ("std", "u1", 255), ("acq", "i2", -32768)):
        a = zarr.open_array(os.path.join(asip, name), mode="w", shape=(nd, n5, n5),
                            chunks=(1, 1024, 1024), dtype=dtype, fill_value=fill)
        for k in range(nd):
            if name == "acq":
                # 15 passes (swaths) as vertical bands, 70 min apart
                band = (np.arange(4096) // 200)[None, :].repeat(4096, 0)
                vals = (band * 70 + 5).astype(np.int16)
                vals[:, 3000:] = -32768
            elif k >= nd - D.W_H:
                vals = rng.integers(0, 101, (4096, 4096)).astype(np.uint8)
                vals[:, 3000:] = 255
            else:  # older days: cheap, compressible fields
                vals = np.full((4096, 4096), 3 * k + (name == "std"), np.uint8)
                vals[:, 2000 + 50 * k:] = 255
            a[(k,) + sl] = vals
    # stored 4 km maps, as written by scripts/convert_sources.py
    sic_a, std_a = (zarr.open_array(os.path.join(asip, n), mode="r") for n in ("sic", "std"))
    fm = D.N_C * 8
    sl_m = (slice(IY0 * 8, IY0 * 8 + fm), slice(IX0 * 8, IX0 * 8 + fm))
    m_arr = {n: zarr.open_array(os.path.join(asip, n), mode="w", shape=(nd, n5 // 8, n5 // 8),
                                chunks=(1, 800, 800), dtype="f4", fill_value=0.0) for n in ("m_mean", "m_w")}
    for k in range(nd):
        m_arr["m_mean"][(k,) + sl_m], m_arr["m_w"][(k,) + sl_m] = asip_to_m(sic_a[(k,) + sl], std_a[(k,) + sl])
    zarr.open_array(os.path.join(asip, "days"), mode="w", shape=(nd,), dtype="i8")[:] = DAYS

    # 5 km grid like the CIMR / CRISTAL / atm files (x: -3847500.., y: 2472500..)
    xc = -3_847_500.0 + 5_000.0 * np.arange(1520)
    yc = 2_472_500.0 - 5_000.0 * np.arange(1475)
    for src, names in D.SOURCES.items():
        path = os.path.join(root, D.STORES[src])
        for name in names:
            a = zarr.open_array(os.path.join(path, name), mode="w", shape=(nd, 1475, 1520),
                                chunks=(8, 512, 512), dtype="f4", fill_value=np.nan)  # as convert_sources.py
            vals = rng.random((nd, 420, 420)).astype(np.float32)  # covers the tile window
            if src == "cristal":
                vals[:, :, ::7] = np.nan  # sparse "tracks"
                vals[:, ::3, :] = np.nan
            a[:, :420, 560:980] = vals
        zarr.open_array(os.path.join(path, "xc"), mode="w", shape=xc.shape, dtype="f8")[:] = xc
        zarr.open_array(os.path.join(path, "yc"), mode="w", shape=yc.shape, dtype="f8")[:] = yc
        zarr.open_array(os.path.join(path, "days"), mode="w", shape=(nd,), dtype="i8")[:] = DAYS


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("stores"))
    _write_stores(root)
    omega_h = np.ones(g.shape("H"), np.float32)
    omega_h[:100, :] = 0.0  # a land strip
    return root, omega_h


def _read(stores, day, horizon=0):
    ds = D.CascadeDataset(*stores[:1], [day], stores[1], np.array([[IY0, IX0]]), train=False, horizon=horizon)
    it = torch.utils.data.default_collate([ds[0]])
    return {k: v[0] for k, v in it.items()}


@pytest.fixture(scope="module")
def raw(stores):
    return _read(stores, DAYS[-1])


@pytest.fixture(scope="module")
def item(raw):
    return D.window(raw, 0)


def test_item_shapes(raw):
    assert raw["asip_sic"].shape == (D.W_H, 4096, 4096) and raw["asip_acq"].shape == (1, 4096, 4096)
    assert raw["asip_m"].shape == raw["asip_m_w"].shape == (D.W - 1, 512, 512)
    assert raw["cimr"].shape == (D.W, 2, D.N5, D.N5) and raw["era5"].shape == (D.W, 4, D.N5, D.N5)
    assert raw["omega_h"].shape == (2048, 2048)
    # rows beyond the store (y > 2475 km) come back as missing
    assert torch.isnan(raw["cimr"][:, :, :100]).all() and torch.isfinite(raw["cimr"][:, :, 200]).all()


def test_horizon_item_windows_match_n0_items(stores, raw):
    seq = _read(stores, DAYS[-3], horizon=2)
    assert seq["asip_sic"].shape[0] == D.W_H + 2 and seq["asip_acq"].shape[0] == 3
    assert seq["asip_m"].shape[0] == D.W + 1 and seq["cimr"].shape[0] == D.W + 2
    last = D.window(seq, 2)
    for k in ("asip_sic", "asip_std", "asip_acq", "asip_m", "asip_m_w", "cimr", "cristal", "era5"):
        assert torch.equal(torch.nan_to_num(last[k], 9.0), torch.nan_to_num(D.window(raw, 0)[k], 9.0)), k
    days = D.build_batch({k: v[None] for k, v in seq.items()})
    assert len(days) == 3 and [int(d["day"]) for d in days] == [DAYS[-3], DAYS[-2], DAYS[-1]]


def test_build_sample_shapes_and_detail_constraint(item):
    out = D.build_sample(item)
    assert out["in_H"].shape == (D.W_H, 2048, 2048)
    assert out["in_C_cimr"].shape == (D.W, 2, 64, 64) and out["f_C"].shape == (D.W, 4, 64, 64)
    assert out["in_M_asip"].shape == (D.W, 512, 512)
    pts = out["pts"]
    assert pts.shape[1] == 5 and len(pts) > 0 and set(pts[:, 3].unique().tolist()) <= {0.0, 1.0}
    assert pts[:, 4].min() >= -(D.W - 1) and pts[:, 4].max() <= 0
    assert out["tgt_C"].shape == (3, 64, 64) and out["tgt_dM"].shape == (2, 512, 512)
    assert out["tgt_dH"].shape == (1, 2048, 2048) and out["tgt_S"].shape == (9, 64, 64)
    assert out["tgt_dH_m"].any() and out["tgt_dM_m"][1].any() and out["tgt_S_m"].any()
    w_h, w_m = out["omega_h"].double(), out["omega_m"].double()
    assert g.restrict(out["tgt_dH"][0].double(), w_h * out["tgt_dH_m"][0], 4)[0].abs().max() < 1e-6
    for v in range(2):
        rd = g.restrict(out["tgt_dM"][v].double(), w_m * out["tgt_dM_m"][v], 8)[0]
        assert rd.abs().max() < 1e-6
    # nothing on land (omega = 0), nothing where there is no data
    for k, f in (("tgt_C_m", 32), ("tgt_M_m", 4), ("tgt_H_m", 1), ("tgt_dM_m", 4), ("tgt_dH_m", 1)):
        assert not out[k][..., : 100 // f, :].any()
    assert not out["tgt_H_m"][0][:, 1500:].any()  # beyond the ASIP swaths


def _perturb(item, where_asip, where_5km):
    it = {k: v.clone() for k, v in item.items()}
    it["asip_sic"][-1][where_asip & (it["asip_sic"][-1] != 255)] = 50
    for src in ("cimr", "cristal"):
        it[src][-1] = torch.where(where_5km, it[src][-1] + 1.0, it[src][-1])
    return it


def test_targets_come_from_y_tgt_only(item):
    part = D.partition(item)
    assert part["asip"].any() and (~part["asip"]).any() and part["blk"].any() and (~part["blk"]).any()
    ref = D.build_sample(item, part)
    # change the day-t input observations: no target may move
    out = D.build_sample(_perturb(item, ~part["asip"], ~part["blk"]), part)
    for k in ref:
        if k.startswith("tgt_"):
            assert torch.equal(out[k], ref[k]), k
    assert not torch.equal(out["in_H"], ref["in_H"])
    # change the day-t target observations: no input may move
    out = D.build_sample(_perturb(item, part["asip"], part["blk"]), part)
    for k in ref:
        if k.startswith("in_") or k == "f_C":
            assert torch.equal(out[k], ref[k]), k
    assert not torch.equal(out["tgt_C"], ref["tgt_C"])


def test_build_batch_stacks_samples(raw):
    batch = {k: torch.stack([v, v]) for k, v in raw.items()}
    (out,) = D.build_batch(batch)
    assert out["tgt_S"].shape == (2, 9, 64, 64) and out["in_H"].shape == (2, D.W_H, 2048, 2048)
    assert out["pts"].shape[0] == 2 and out["pts_valid"].any(1).all()


def test_worker_asip_m_equals_restricted_h_gridding(item):
    # day t-1 is both in the H window (GPU gridding) and aggregated at M by the worker
    out = D.build_sample(item)
    m_gpu, w_gpu = g.restrict(out["in_H"][-2].double(), out["in_H_w"][-2].double(), 4)
    assert torch.allclose(item["asip_m"][-1].double(), m_gpu, atol=1e-5)
    assert torch.allclose(item["asip_m_w"][-1].double(), w_gpu, rtol=1e-5)
    # older days only exist at M, with their own coverage
    assert item["asip_m_w"][0].gt(0).any() and torch.allclose(item["asip_m"][0][item["asip_m_w"][0] > 0], torch.tensor(0.0))
    assert item["asip_m"][1][item["asip_m_w"][1] > 0].sub(0.03).abs().max() < 1e-6


def test_asip_pass_crossing_a_bin_edge_is_one_unit():
    acq = torch.full((40, 40), -32768, dtype=torch.int16)
    acq[:, :10] = 295            # pass 1, scenes from 4:55 to 5:05 (crosses 5:00)
    acq[:, 10:20] = 297
    acq[:, 20:25] = 303
    acq[:, 25:30] = 305
    acq[:, 30:] = 410            # pass 2, 105 min later
    valid = acq != -32768
    unit = D.asip_passes(acq, valid)
    assert len(unit[:, :30].unique()) == 1 and len(unit[:, 30:].unique()) == 1
    assert unit[0, 0] != unit[0, 35]
    for seed in range(20):
        tgt = obs.unit_partition(unit, valid, 0.5, torch.Generator().manual_seed(seed))
        assert tgt[:, :30].all() or not tgt[:, :30].any()
