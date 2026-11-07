"""
Configuration of the realism-scenario sweep, shared by qg2l_realism_sweep.ipynb and run_sweep.py.

Method / generation dictionaries are kept identical across versions so that the configuration hashes
(and hence all cached results) stay valid; only *new* entries trigger new computations.
"""
from pathlib import Path

from . import obs as O
from .simulate import PRESETS


def config(size="small"):
    preset = "energetic"
    if size == "small":
        L, nx, nx_truth, spinup = 5e5, 32, 64, 300
        splits = dict(train=[0, 1095], val=[1095, 1185], test=[1215, 1305], total=1395)
        net = dict(n_time=21, epochs=120)
        lyap = dict(k=192, n_fwd=30, n_bwd=60, animation_days=30, figure_methods=None)
        window, spin_var = 7, 30
        run = ["baseline", "obs_signal", "param_error", "B_free_run", "realistic"]
        seed_scenarios = ["baseline", "realistic"]
        tune = dict(grid_b=(0.25, 0.5, 1.0), grid_w=(5, 7, 10), grid_q=(0.05, 0.2, 0.5), n_iter=20, r_mode="oracle")
    elif size == "full":
        L, nx, nx_truth, spinup = 1e6, 64, 128, 1095
        splits = dict(train=[0, 1095], val=[1095, 1277], test=[1307, 1672], total=1762)
        net = dict(n_time=21, epochs=150)
        lyap = dict(k=512, n_fwd=30, n_bwd=60, animation_days=60, figure_methods=None)
        window, spin_var = 10, 60
        run = ["baseline", "realistic"]  # final confirmation at the paper resolution
        seed_scenarios = ["baseline"]
        tune = dict(grid_b=(0.5, 1.0), grid_w=(7, 10), grid_q=(0.2,), n_iter=20, r_mode="oracle")
    else:
        raise ValueError(size)
    seeds = [333, 1, 2]  # 333 = reference seed ('4dvarnet'), others -> '4dvarnet_s<seed>'

    # constellation with the real ground-track spacing whatever the box size
    const = {n: dict(s, tracks_per_dir=max(1, round(s["tracks_per_dir"] * L / 1e6)))
             for n, s in O.DEFAULT_CONSTELLATION.items()}
    base_gen = dict(preset=preset, nx=nx, nx_truth=nx_truth, L=L, spinup_days=spinup, splits=splits,
                    constellation=const, animation_days=30)

    common = dict(window_days=window, b_scale=0.5, spinup_days=spin_var)
    methods = [
        dict(name="4dvar_strong", mode="strong", solver="lbfgs", n_iter=40, **common),
        dict(name="4dvar_weak", mode="weak", solver="lbfgs", n_iter=40, q_scale=0.2, **common),
        dict(name="4dvar_incr", mode="strong", solver="incremental", n_outer=3, n_inner=20, **common),
        dict(name="4dvar_incr_budget", mode="strong", solver="incremental", n_outer=1, n_inner=5, **common),
        dict(name="hybrid", mode="strong", solver="incremental", n_outer=1, n_inner=10, window_days=window,
             b_scale=0.2, spinup_days=0, background="4dvarnet"),
        dict(name="4dvar_R_desroziers", mode="strong", solver="lbfgs", n_iter=40, r_mode="desroziers", **common),
        dict(name="4dvar_R_oracle", mode="strong", solver="lbfgs", n_iter=40, r_mode="oracle", **common),
        dict(name="hybrid_w3", mode="strong", solver="incremental", n_outer=1, n_inner=10, window_days=3,
             b_scale=0.2, spinup_days=0, background="4dvarnet"),
        dict(name="hybrid_b01", mode="strong", solver="incremental", n_outer=1, n_inner=10, window_days=window,
             b_scale=0.1, spinup_days=0, background="4dvarnet"),
        dict(name="hybrid_weak", mode="weak", solver="incremental", n_outer=1, n_inner=10, window_days=window,
             b_scale=0.2, q_scale=0.2, spinup_days=0, background="4dvarnet"),
        dict(name="hybrid_w3_Rdes", mode="strong", solver="incremental", n_outer=1, n_inner=10, window_days=3,
             b_scale=0.2, spinup_days=0, background="4dvarnet", r_mode="desroziers"),
    ]

    P = PRESETS[preset]
    signal = dict(ssh_std=0.01, wavelength_km=(40.0, 120.0), decorrelation_days=1.0)
    const_roll = {n: dict(s, roll_std=0.01 if s["kind"] == "swot" else 0.0) for n, s in const.items()}
    const_nadir = {n: s for n, s in const.items() if s["kind"] == "nadir"}
    params = dict(rek=1.5 * P["rek"], beta=0.8 * P["beta"])
    forcing = dict(ssh_std=0.002, wavelength_km=(150.0, 400.0))
    scenarios = {
        "baseline": dict(gen={}),
        "obs_hr": dict(gen=dict(obs_source="hr_block")),
        "obs_signal": dict(gen=dict(obs_source="hr_block", unresolved_signal=signal)),
        "swot_roll": dict(gen=dict(constellation=const_roll)),
        "param_error": dict(gen=dict(truth_overrides=params)),
        "forcing": dict(gen=dict(forcing=forcing)),
        "B_free_run": dict(gen={}, var=dict(b_source="free_run")),
        "nadir_only": dict(gen=dict(constellation=const_nadir)),
        "realistic": dict(gen=dict(obs_source="hr_block", unresolved_signal=signal, constellation=const_roll,
                                   truth_overrides=params, forcing=forcing), var=dict(b_source="free_run")),
    }
    fig_methods = ["4dvarnet", "4dvar_tuned", "4dvar_R_desroziers", "hybrid_w3_Rdes"]
    lyap["figure_methods"] = fig_methods
    method_order = ["4dvarnet", "4dvarnet_s1", "4dvarnet_s2", "4dvar_tuned", "4dvar_tuned_weak", "4dvar_strong",
                    "4dvar_weak", "4dvar_incr", "4dvar_incr_budget", "4dvar_R_desroziers", "4dvar_R_oracle",
                    "hybrid", "hybrid_w3", "hybrid_b01", "hybrid_weak", "hybrid_w3_Rdes"]
    return dict(SIZE=size, ROOT=Path("data/qg2l_scenarios_" + size), BASE_GEN=base_gen, NET=net, LYAP=lyap,
                WINDOW=window, SPIN_VAR=spin_var, METHODS=methods, SCENARIOS=scenarios, RUN=run, SEEDS=seeds,
                SEED_SCENARIOS=seed_scenarios, TUNE=tune, FIG_METHODS=fig_methods, METHOD_ORDER=method_order)


def tuned_methods(tuning, spin_var, var_over):
    """4DVar with the hyper-parameters selected on the validation split and the Desroziers R."""
    out = []
    if tuning and tuning.get("best_strong"):
        out.append(dict(name="4dvar_tuned", mode="strong", solver="lbfgs", n_iter=40, spinup_days=spin_var,
                        r_mode="desroziers", **tuning["best_strong"], **var_over))
    if tuning and tuning.get("best_weak"):
        out.append(dict(name="4dvar_tuned_weak", mode="weak", solver="lbfgs", n_iter=40, spinup_days=spin_var,
                        r_mode="desroziers", **tuning["best_weak"], **var_over))
    return out
