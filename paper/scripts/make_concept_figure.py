"""
Conceptual figure of the paper (figures/concept.pdf): stable / neutral / unstable subspaces on a simple
chaotic system (Lorenz-63, covariant Lyapunov vectors computed with the Ginelli algorithm), how an error
cloud deforms, the Lyapunov-subspace decomposition used by the diagnostic, and the mechanism compared
(dynamically consistent 4DVar vs learned smoother). numpy + matplotlib only.

    python paper/scripts/make_concept_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures"
SIG, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
C_U, C_N, C_S = "#d62728", "#7f7f7f", "#1f77b4"


def f(x):
    return np.array([SIG * (x[1] - x[0]), x[0] * (RHO - x[2]) - x[1], x[0] * x[1] - BETA * x[2]])


def jac(x):
    return np.array([[-SIG, SIG, 0.0], [RHO - x[2], -1.0, -x[0]], [x[1], x[0], -BETA]])


def rk4(x, dt):
    k1 = f(x)
    k2 = f(x + dt / 2 * k1)
    k3 = f(x + dt / 2 * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def tlm_step(x, V, dt):
    """RK4 tangent linear step of the 3 perturbations in V (columns)."""
    J = jac
    k1 = J(x) @ V
    x2 = x + dt / 2 * f(x)
    k2 = J(x2) @ (V + dt / 2 * k1)
    x3 = x + dt / 2 * f(x2)
    k3 = J(x3) @ (V + dt / 2 * k2)
    x4 = x + dt * f(x3)
    k4 = J(x4) @ (V + dt * k3)
    return V + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz_clv(n_steps=6000, dt=0.01, n_spin=1000, n_tr=1500):
    x = np.array([1.0, 1.0, 20.0])
    for _ in range(n_spin):
        x = rk4(x, dt)
    traj, Qs, Rs = [], [], []
    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))[0]
    logs = []
    for n in range(n_steps):
        traj.append(x)
        Qs.append(Q)
        W = tlm_step(x, Q, dt)
        Q, R = np.linalg.qr(W)
        s = np.sign(np.diag(R))
        Q, R = Q * s, R * s[:, None]
        Rs.append(R)
        logs.append(np.log(np.abs(np.diag(R))))
        x = rk4(x, dt)
    lam = np.mean(logs[n_tr:], 0) / dt
    # Ginelli backward pass
    C = np.triu(np.ones((3, 3)))
    C /= np.linalg.norm(C, axis=0)
    clv = [None] * n_steps
    for n in range(n_steps - 1, -1, -1):
        clv[n] = Qs[n] @ C
        C = np.linalg.solve(Rs[n - 1], C) if n > 0 else C
        C /= np.linalg.norm(C, axis=0)
    return np.array(traj), np.array(clv), lam


def arrow(ax, p, v, color, scale, lw=1.6):
    ax.add_patch(FancyArrowPatch(p, p + scale * v, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw))


def main():
    traj, clv, lam = lorenz_clv()
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(7.2, 5.2), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    # (a) attractor + CLVs (x-z projection) ----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    P = [0, 2]
    seg = traj[1500:5500]
    ax.plot(seg[:, 0], seg[:, 2], color="0.75", lw=0.4)
    for n in [1700, 2240, 2900, 3550, 4300, 5050]:
        p = traj[n][P]
        for j, col in [(0, C_U), (1, C_N), (2, C_S)]:
            v = clv[n][P, j]
            v = v / max(np.linalg.norm(v), 1e-9)
            arrow(ax, p, v, col, 6.5)
        ax.plot(*p, "k.", ms=3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_title("(a) covariant Lyapunov vectors along the flow", loc="left")
    for col, lab, l in [(C_U, "unstable", lam[0]), (C_N, "neutral (along the flow)", lam[1]),
                        (C_S, "stable", lam[2])]:
        ax.plot([], [], color=col, lw=2, label=f"{lab}, $\\lambda$={l:.2f}")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)

    # (b) error cloud in the local Lyapunov frame (tangent-linear evolution) ------------------
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(1)
    n0, dt = 2040, 0.01  # a point where the finite-time unstable growth is positive (as on average)
    th = np.linspace(0, 2 * np.pi, 400)
    a0 = np.stack([np.cos(th), 0 * th, np.sin(th)])  # unit circle in the (unstable, stable) CLV plane
    D = clv[n0] @ a0  # corresponding physical perturbations
    cols = ["#444444", "#9467bd", "#e377c2"]
    labels = {0: "$t=0$", 10: "$t=0.1$", 30: "$t=0.3$"}
    x, k = traj[n0].copy(), 0
    for step in range(31):
        if step in labels:
            a = np.linalg.solve(clv[n0 + step], D)  # coordinates in the local CLV basis
            ax.plot(a[0], a[2], color=cols[k], lw=1.6, label=labels[step])
            k += 1
        D = tlm_step(x, D, dt)
        x = rk4(x, dt)
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.axvline(0, color="0.8", lw=0.6, zorder=0)
    ax.annotate("", xy=(3.4, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_U, lw=1.6))
    ax.annotate("", xy=(0, 1.15), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_S, lw=1.6))
    ax.text(3.0, 0.15, "unstable\nCLV", color=C_U, fontsize=7)
    ax.text(0.08, 1.12, "stable CLV", color=C_S, fontsize=7)
    ax.set_xlim(-3.4, 4.0)
    ax.set_ylim(-1.3, 1.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("coordinate on the unstable CLV")
    ax.set_ylabel("coordinate on the stable CLV")
    ax.set_title("(b) error cloud in the local Lyapunov frame", loc="left")
    ax.legend(loc="lower left", fontsize=6.5, frameon=False)

    # (c) Lyapunov spectrum of a large system + diagnostic subspaces ---------------------------
    ax = fig.add_subplot(gs[1, 0])
    i = np.arange(1, 241)
    spec = 0.14 * np.exp(-i / 60) - 0.046 + 0.002 * np.sin(i / 3)
    n0, nn, k = int((spec > 0.01).sum()), int((spec > -0.01).sum()), 192
    colors = np.where(spec > 0.01, C_U, np.where(spec > -0.01, C_N, C_S))
    ax.bar(i[:k], spec[:k], color=colors[:k], width=1.0)
    ax.bar(i[k:], spec[k:], color="0.85", width=1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(k + 0.5, color="k", lw=0.8, ls="--")
    ax.text(4, 0.112, "unstable: error grows", color=C_U, ha="left", va="bottom")
    ax.text((n0 + nn) / 2, 0.022, "neutral", color=C_N, ha="center")
    ax.text((nn + k) / 2 - 8, -0.066, "stable (resolved): damped", color=C_S, ha="center")
    ax.text(k + 24, 0.03, "residual\n(most stable,\nbeyond $k$)", color="0.4", ha="center", fontsize=7)
    ax.annotate("$n_0$", xy=(n0 + 0.5, 0.0), xytext=(n0 + 14, 0.065), arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.annotate("$k$", xy=(k + 0.5, 0.11), xytext=(k - 18, 0.11), arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("Lyapunov index $i$")
    ax.set_ylabel("$\\lambda_i$ (day$^{-1}$)")
    ax.set_ylim(-0.075, 0.13)
    ax.set_title("(c) Lyapunov spectrum and error decomposition", loc="left")

    # (d) mechanism: within-window growth vs learned smoother ------------------------------
    ax = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 28, 1400)
    W, lam1 = 7.0, 0.11
    tw = np.mod(t, W)
    var_u = 0.25 * np.exp(2 * lam1 * tw)
    var_s = 0.55 + 0.25 * np.exp(-0.6 * tw)
    net_u = 0.12 + 0.015 * np.sin(2 * np.pi * t / 5.3)
    net_s = 0.65 + 0.02 * np.sin(2 * np.pi * t / 4.1)
    ax.plot(t, var_u, color=C_U, lw=1.6, label="4DVar, unstable part")
    ax.plot(t, var_s, color=C_S, lw=1.6, label="4DVar, stable part")
    ax.plot(t, net_u, color=C_U, lw=1.6, ls="--", label="learned smoother, unstable part")
    ax.plot(t, net_s, color=C_S, lw=1.6, ls="--", label="learned smoother, stable part")
    for w in np.arange(0, 28.1, W):
        ax.axvline(w, color="0.85", lw=0.8, zorder=0)
    ax.text(W / 2, 1.27, "window 1", ha="center", color="0.4", fontsize=7)
    ax.text(1.5 * W, 1.27, "window 2", ha="center", color="0.4", fontsize=7)
    ax.annotate("analysis = model trajectory:\nunstable error grows $\\propto e^{2\\lambda_1 t}$",
                xy=(20.5, 0.25 * np.exp(2 * lam1 * 6.5)), xytext=(15.2, 1.32), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("error energy (schematic)")
    ax.set_title("(d) 4DVar vs learned smoother", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.5, frameon=False, ncol=2)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "concept.pdf")
    fig.savefig(OUT / "concept.png", dpi=200)
    print("Lorenz-63 exponents:", np.round(lam, 3))


if __name__ == "__main__":
    main()
