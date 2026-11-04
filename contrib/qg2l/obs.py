"""
Synthetic satellite altimetry sampling on the doubly periodic QG domain.

Each satellite has a repeat cycle (days), a number of ground tracks per direction
crossing the box during one cycle, and a heading angle from the meridian at the
box latitude (sin a = cos(inclination) / cos(lat), Earth rotation neglected).
Ascending and descending tracks have headings +a and -a. Tracks of one cycle are
spread over the cycle days with an interleaved sub-cycle, as on real repeat orbits.

* nadir altimeters: one-pixel line along the track
* SWOT (KaRIn): two swaths of ``swath_width`` km separated by a ``nadir_gap`` km gap
  (+ optional nadir altimeter line inside the gap)

Noise: white noise of std ``noise`` (m) per pixel, plus for SWOT an optional
roll-like error linear in the cross-track distance, random per pass.
All sampling is done at daily snapshots (an observation at day d samples the
true field of day d).
"""
import math
import numpy as np


DEFAULT_CONSTELLATION = {
    # name: dict(kind, repeat_days, tracks_per_dir, inclination_deg, noise_m, ...)
    "jason3": dict(kind="nadir", repeat_days=10, tracks_per_dir=4, inclination=66.0, noise=0.01, phase=0),
    "sentinel3a": dict(kind="nadir", repeat_days=27, tracks_per_dir=10, inclination=98.65, noise=0.01, phase=3),
    "sentinel3b": dict(kind="nadir", repeat_days=27, tracks_per_dir=10, inclination=98.65, noise=0.01, phase=16),
    "swot": dict(kind="swot", repeat_days=21, tracks_per_dir=9, inclination=77.6, noise=0.003,
                 swath_width=50.0, nadir_gap=20.0, roll_std=0.0, with_nadir=True, phase=5),
}


def heading_angle(inclination_deg, lat_deg):
    """Ground-track heading from the meridian (rad) at a given latitude."""
    s = math.cos(math.radians(inclination_deg)) / math.cos(math.radians(lat_deg))
    return math.asin(max(-1.0, min(1.0, s)))


def track_schedule(repeat_days, tracks_per_dir, seed_phase=0):
    """List per cycle day of (direction sign, track index) observed that day."""
    n = 2 * tracks_per_dir
    days = [[] for _ in range(repeat_days)]
    # interleave so that successive passes are far apart (sub-cycle like)
    step = max(1, int(round(repeat_days / max(1, n) * 3))) or 1
    for j in range(n):
        d = (seed_phase + j * step + (j // max(1, math.gcd(step, repeat_days)))) % repeat_days
        sgn = 1 if j % 2 == 0 else -1
        days[d].append((sgn, j // 2))
    return days


def _signed_distance(X, Y, x0, angle, L):
    """Signed periodic distance (m) from points to the line through (x0, 0) with heading angle."""
    # line direction (sin a, cos a); normal (cos a, -sin a)
    d = (X - x0) * math.cos(angle) - Y * math.sin(angle)
    period = L * abs(math.cos(angle)) if abs(math.cos(angle)) > 1e-6 else L
    return (d + 0.5 * period) % period - 0.5 * period


def daily_masks(sat, n_days, nx, L, lat=35.0, day0=0, rng=None):
    """Observation masks / noise for one satellite.

    Returns obs_mask (n_days, nx, nx) bool, noise (n_days, nx, nx) float (m),
    and track_id (n_days, nx, nx) int (-1 where unobserved).
    """
    rng = rng or np.random.default_rng(0)
    dx = L / nx
    x = (np.arange(nx) + 0.5) * dx
    X, Y = np.meshgrid(x, x)
    a = heading_angle(sat["inclination"], lat)
    sched = track_schedule(sat["repeat_days"], sat["tracks_per_dir"], sat.get("phase", 0))
    spacing = L / sat["tracks_per_dir"]
    mask = np.zeros((n_days, nx, nx), bool)
    noise = np.zeros((n_days, nx, nx))
    tid = -np.ones((n_days, nx, nx), int)
    for t in range(n_days):
        cyc_day = (day0 + t) % sat["repeat_days"]
        for sgn, j in sched[cyc_day]:
            x0 = (j + (0.25 if sgn > 0 else 0.75)) * spacing
            d = _signed_distance(X, Y, x0, sgn * a, L)
            if sat["kind"] == "nadir":
                m = np.abs(d) < 0.5 * dx
            else:
                half_gap = 0.5 * sat["nadir_gap"] * 1e3
                w = sat["swath_width"] * 1e3
                m = (np.abs(d) >= half_gap) & (np.abs(d) < half_gap + w)
                if sat.get("with_nadir", False):
                    m |= np.abs(d) < 0.5 * dx
            n = sat["noise"] * rng.standard_normal((nx, nx))
            if sat["kind"] == "swot" and sat.get("roll_std", 0.0) > 0:
                n = n + sat["roll_std"] * rng.standard_normal() * d / 1e5  # per 100 km
            new = m & ~mask[t]
            mask[t] |= m
            noise[t][new] = n[new]
            tid[t][new] = 1000 * (day0 + t) + 10 * j + (sgn > 0)
    return mask, noise, tid


def build_observations(ssh, L, constellation=None, lat=35.0, seed=0, day0=0):
    """Sample daily ssh fields (n_days, nx, nx) with a constellation.

    Returns dict with
        obs        (n_days, nx, nx)  noisy ssh, NaN where unobserved (all satellites merged)
        obs_sigma  (n_days, nx, nx)  obs error std (NaN where unobserved)
        sat_id     (n_days, nx, nx)  index of the satellite in ``names`` (-1 unobserved)
        names      list of satellite names
    When two satellites observe the same pixel the same day, the first one listed wins.
    """
    constellation = constellation or DEFAULT_CONSTELLATION
    rng = np.random.default_rng(seed)
    n_days, nx, _ = ssh.shape
    obs = np.full(ssh.shape, np.nan)
    sig = np.full(ssh.shape, np.nan)
    sid = -np.ones(ssh.shape, int)
    names = list(constellation)
    for i, name in enumerate(names):
        sat = constellation[name]
        mask, noise, _ = daily_masks(sat, n_days, nx, L, lat=lat, day0=day0, rng=rng)
        new = mask & (sid < 0)
        obs[new] = ssh[new] + noise[new]
        sig[new] = sat["noise"]
        sid[new] = i
    return dict(obs=obs, obs_sigma=sig, sat_id=sid, names=names)


def coverage_stats(sat_id, names):
    n = sat_id[0].size
    return {name: float((sat_id == i).sum() / sat_id.shape[0] / n) for i, name in enumerate(names)}


def band_limited_field(n, L, wavelength_km, rng):
    """Unit-variance random field on the periodic (n, n) grid with wavelengths in [lmin, lmax] km."""
    lmin, lmax = wavelength_km
    k = np.fft.fftfreq(n, L / n)  # cycles / m
    kk = np.sqrt(k[None, :] ** 2 + k[:, None] ** 2)
    band = (kk >= 1e-3 / lmax) & (kk <= 1e-3 / lmin)
    if not band.any():
        raise ValueError("no resolved wavenumber in the requested band")
    fld = np.fft.ifft2(band * np.fft.fft2(rng.standard_normal((n, n)))).real
    return fld / fld.std()


def unresolved_signal(n_days, nx, L, ssh_std, wavelength_km=(50.0, 150.0), decorrelation_days=1.0, rng=None):
    """SSH signal seen by the altimeters but absent from the truth / model (internal-tide-like):
    band-limited in space, AR(1) in time with the given decorrelation time. (n_days, nx, nx) in m."""
    rng = rng or np.random.default_rng(0)
    a = math.exp(-1.0 / decorrelation_days) if decorrelation_days > 0 else 0.0
    out = np.empty((n_days, nx, nx))
    s = band_limited_field(nx, L, wavelength_km, rng)
    for t in range(n_days):
        if t:
            s = a * s + math.sqrt(1 - a**2) * band_limited_field(nx, L, wavelength_km, rng)
        out[t] = ssh_std * s
    return out
