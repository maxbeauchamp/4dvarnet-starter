#!/usr/bin/env python3
"""Agrège les batches preproc par résolution en un seul NetCDF compressé.

Remplace ncecat (NCO 5.0.6 segfaulte au-delà de quelques centaines de fichiers).
xarray+dask écrit le fichier en streaming chunk par chunk, sans jamais
matérialiser une variable entière en mémoire.

Usage : python aggregate_xarray.py <res>     # ex. 2, 10, 50
"""
import glob
import re
import sys

import xarray as xr

INPUT_DIR = "/dmidata/users/maxb/PREPROC"
PREFIX = "preproc_batch"
OUTPUT_PREFIX = "preproc_CROSCIM"


def main(res: str) -> None:
    pattern = f"{INPUT_DIR}/{PREFIX}_*_x{res}.nc"
    files = glob.glob(pattern)
    files.sort(key=lambda f: int(re.search(rf"{PREFIX}_(\d+)_x{res}\.nc$", f).group(1)))
    if not files:
        sys.exit(f"Aucun fichier pour x{res} ({pattern})")
    print(f"x{res} : {len(files)} fichiers", flush=True)

    # Chaque fichier a sample=1+ ; on les empile le long de 'sample'.
    ds = xr.open_mfdataset(
        files,
        concat_dim="sample",
        combine="nested",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=False,
    )
    ds = ds.chunk({"sample": 50})

    # Chunk HDF5 = 1 sample (le reste plein) : aligné sur l'écriture sample-par-sample,
    # sinon HDF5 recompresse le même chunk des centaines de fois (~150x plus lent).
    enc = {}
    for v in ds.data_vars:
        if "sample" in ds[v].dims:
            cs = tuple(1 if d == "sample" else ds.sizes[d] for d in ds[v].dims)
            enc[v] = dict(zlib=True, complevel=1, chunksizes=cs)
        else:
            enc[v] = dict(zlib=True, complevel=1)

    out = f"{INPUT_DIR}/{OUTPUT_PREFIX}_x{res}.nc"
    print(f"Écriture -> {out}", flush=True)
    ds.to_netcdf(out, encoding=enc, engine="netcdf4")
    print(f"✅ Terminé : {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python aggregate_xarray.py <res>")
    main(sys.argv[1])
