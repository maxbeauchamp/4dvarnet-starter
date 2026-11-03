"""
Figures and animations for the QG testbed. Functions accept an xarray.Dataset
produced by simulate.to_dataset (or any mapping with the same variable names and
an ``attrs`` dict holding ``model_kw``).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

from .qgmodel import QG2L, isotropic_spectrum  # noqa: E402

SAT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]


def _arr(ds, k):
    v = ds[k]
    return np.asarray(getattr(v, "values", v))


def _model(ds):
    return QG2L(**json.loads(ds.attrs["model_kw"]))


def _extent(ds):
    L = json.loads(ds.attrs["model_kw"])["L"] / 1e3
    return [0, L, 0, L]


def _sym(a, q=99.5):
    v = np.nanpercentile(np.abs(a), q)
    return dict(vmin=-v, vmax=v)


def plot_snapshot(ds, day, path=None):
    m = _model(ds)
    psi = _arr(ds, "psi")[day].astype(np.float64)
    ssh = m.ssh(psi)
    zeta = m.vorticity(psi) / m.f0
    obs = _arr(ds, "obs_ssh")[day]
    ext = _extent(ds)
    f, ax = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    panels = [
        (ssh, "SSH truth [m]", "RdBu_r", _sym(ssh)),
        (zeta[0], r"$\zeta_1 / f_0$", "RdBu_r", _sym(zeta[0])),
        (zeta[1], r"$\zeta_2 / f_0$", "RdBu_r", _sym(zeta[0])),
        (obs, f"SSH obs, day {day} [m]", "RdBu_r", _sym(ssh)),
        (psi[0], r"$\psi_1$ [m$^2$/s]", "PuOr_r", _sym(psi[0])),
        (psi[1], r"$\psi_2$ [m$^2$/s]", "PuOr_r", _sym(psi[0])),
    ]
    for a, (fld, title, cmap, lim) in zip(ax.ravel(), panels):
        im = a.imshow(fld, origin="lower", extent=ext, cmap=cmap, **lim)
        a.set_title(title)
        a.set_xlabel("x [km]")
        plt.colorbar(im, ax=a, shrink=0.85)
    if path:
        f.savefig(path, dpi=110)
        plt.close(f)
    return f


def plot_spectra(ds, path=None, days=None):
    m = _model(ds)
    psi = _arr(ds, "psi")
    days = days if days is not None else slice(None, None, 10)
    sel = psi[days].astype(np.float64)
    f, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    k, e_ssh = isotropic_spectrum(m.ssh(sel), m.dx)
    ax[0].loglog(k / (2 * np.pi) * 1e3, e_ssh, "k", label="SSH truth (DA grid)")
    kref = k[3:len(k) // 2]
    ax[0].loglog(kref / (2 * np.pi) * 1e3, e_ssh[3] * (kref / kref[0]) ** -5, "--", c="gray", label=r"$k^{-5}$")
    ax[0].set_xlabel("wavenumber [cycles/km]")
    ax[0].set_ylabel("PSD SSH")
    ax[0].axvline(1 / (2 * np.pi * m.rd / 1e3), c="tab:red", lw=0.8, label=r"$1/(2\pi R_d)$")
    ax[0].legend()
    for i, c in zip(range(2), ["tab:blue", "tab:orange"]):
        _, ke = isotropic_spectrum(np.stack(m.velocity(sel[:, i])), m.dx)
        ax[1].loglog(k / (2 * np.pi) * 1e3, ke, c=c, label=f"KE layer {i+1}")
    ax[1].set_xlabel("wavenumber [cycles/km]")
    ax[1].legend()
    if path:
        f.savefig(path, dpi=110)
        plt.close(f)
    return f


def plot_coverage(ds, path=None, days=21, revisit_map=None):
    sid = _arr(ds, "sat_id")
    names = json.loads(ds.attrs["sat_names"])
    ext = _extent(ds)
    ncols = 4
    f, ax = plt.subplots(1, ncols, figsize=(17, 4.4), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(SAT_COLORS[: len(names)])
    for j, d in enumerate([0, 1, 2]):
        a = ax[j]
        a.imshow(np.where(sid[d] >= 0, sid[d], np.nan), origin="lower", extent=ext, cmap=cmap,
                 vmin=-0.5, vmax=len(names) - 0.5, interpolation="nearest")
        a.set_title(f"observations day {d}")
    handles = [plt.Line2D([], [], c=SAT_COLORS[i], lw=4, label=n) for i, n in enumerate(names)]
    ax[0].legend(handles=handles, loc="upper left", fontsize=8)
    if revisit_map is None:
        seen = sid >= 0
        revisit_map = np.where(seen.sum(0) > 0, seen.shape[0] / np.maximum(seen.sum(0), 1), np.nan)
    im = ax[3].imshow(np.where(np.isfinite(revisit_map), revisit_map, np.nan), origin="lower", extent=ext,
                      cmap="viridis_r", vmax=np.nanpercentile(revisit_map[np.isfinite(revisit_map)], 98))
    ax[3].set_title(f"mean revisit time [days] (median {np.nanmedian(revisit_map[np.isfinite(revisit_map)]):.1f})")
    plt.colorbar(im, ax=ax[3], shrink=0.85)
    if path:
        f.savefig(path, dpi=110)
        plt.close(f)
    return f


def plot_energy(energy, path=None):
    f, a = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    a.plot(np.arange(len(energy)), energy)
    a.set_xlabel("day (after spin-up)")
    a.set_ylabel(r"total energy [m$^2$ s$^{-2}$]")
    if path:
        f.savefig(path, dpi=110)
        plt.close(f)
    return f


def dataset_figures(ds, fig_dir, energy=None, revisit_map=None):
    fig_dir = Path(fig_dir)
    splits = json.loads(ds.attrs["splits"])
    t0 = splits["test"][0]
    plot_snapshot(ds, t0, fig_dir / "snapshot_test_day0.png")
    plot_spectra(ds, fig_dir / "spectra.png")
    plot_coverage(ds, fig_dir / "obs_coverage.png", revisit_map=revisit_map)
    if energy is not None:
        plot_energy(energy, fig_dir / "energy.png")


def animate(ds, path, days, fields=None, fps=8):
    """Animation of truth SSH, obs, truth zeta_1, psi_2 (and optional extra fields).

    fields: optional dict name -> array (n_time_total, ny, nx) added as extra panels
            (e.g. reconstructions). Saved as mp4 when ffmpeg is available, gif otherwise.
    """
    m = _model(ds)
    days = list(days)
    psi = _arr(ds, "psi")[days].astype(np.float64)
    ssh = m.ssh(psi)
    zeta = m.vorticity(psi)[:, 0] / m.f0
    obs = _arr(ds, "obs_ssh")[days]
    panels = [("SSH truth [m]", ssh, "RdBu_r", _sym(ssh)), ("SSH obs [m]", obs, "RdBu_r", _sym(ssh)),
              (r"$\zeta_1/f_0$", zeta, "RdBu_r", _sym(zeta)), (r"$\psi_2$", psi[:, 1], "PuOr_r", _sym(psi[:, 1]))]
    for name, arr in (fields or {}).items():
        a = np.asarray(arr)[days]
        own_scale = "err" in name.lower() or a.ndim != 3
        panels.append((name, a, "RdBu_r", _sym(a) if own_scale else _sym(ssh)))
    ext = _extent(ds)
    n = len(panels)
    f, ax = plt.subplots(1, n, figsize=(3.6 * n, 3.7), constrained_layout=True)
    ims = []
    for a, (title, arr, cmap, lim) in zip(np.atleast_1d(ax), panels):
        ims.append(a.imshow(arr[0], origin="lower", extent=ext, cmap=cmap, **lim))
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])
    sup = f.suptitle(f"day {days[0]}")

    def update(i):
        for im, (_, arr, _, _) in zip(ims, panels):
            im.set_data(arr[i])
        sup.set_text(f"day {days[i]}")
        return ims

    anim = animation.FuncAnimation(f, update, frames=len(days), interval=1000 / fps, blit=False)
    path = Path(path)
    if path.suffix == ".mp4" and animation.writers.is_available("ffmpeg"):
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    else:
        path = path.with_suffix(".gif")
        anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(f)
    return path
