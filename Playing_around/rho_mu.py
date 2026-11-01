"""Density of states and thermodynamics at non-zero baryon chemical potential.

    d log rho_0 / dc = P ( <F>_c - c ),        P = 2e4
    rho(c, mu) = rho_0(c) * exp( V * sum_n B_n(c) mu^n / n! ),   n = 2, 4, 6, 8

with mu = mu_B / T and V = V T^3 = (N_s / N_t)^3 = (16 / 8)^3 = 8.

From rho(c, mu) we compute
    <c>, width = sqrt(<c^2> - <c>^2)
    pressure   Delta p / T^4 = (1/V) log[ Z(mu) / Z(0) ],
               Z(mu) = int rho_0(c) exp( V sum_n B_n mu^n / n! ) dc
    density    n_B / T^3 = d(Delta p / T^4)/d mu = < sum_n B_n(c) mu^(n-1) / (n-1)! >_mu

Input: raw/R.dat (= <F>_c) and raw/B{n}.dat, copied from
katinka:/data/4hex/ana_dos/16x8/150. Columns: c, mean, error, 48 jackknife samples.
Each c value is a separate simulation, so the points are independent and the
chi^2 of a fit in c is meaningful.

Procedure
  1. Fit a polynomial in c to d log rho_0/dc and to each B_n, on the mean and on
     each jackknife sample (same 1/error weights for all samples).
  2. Integrate, add the B_n terms, compute the observables, for the mean and for
     each sample.
  3. Repeat 1-2 for several fit variants (fit range x polynomial degree) and
     combine them by AIC model averaging:
         AIC_v = sum over fitted quantities of [chi^2 + 2 n_params + 2 n_cut],
         n_cut = number of data points left out by the fit range,
         weight w_v ~ exp(-AIC_v / 2).
     central = sum_v w_v x_v,  sys^2 = sum_v w_v (x_v - central)^2.
     stat = jackknife error of the weighted average (weights fixed from the means).
     total error = sqrt(stat^2 + sys^2).
"""
from math import factorial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as poly

P = 2e4
V = 8.0
N_JK = 48
ORDERS = [2, 4, 6, 8]
RANGES = [(-0.2, 0.9), (-0.26, 1.0)]  # fit ranges in c
DEGREE_OFFSETS = [-1, 0, 1, 2, 3]     # degrees tried: AIC-best + these
MAX_DEGREE = 10
DC = 0.005                            # c spacing of the grid the fits are evaluated on
MU_PLOT = [0.0, 1.0, 1.5, 2.0, 2.5, 6.0]   # mu_B / T values for the rho(c, mu) plots
MU_SCAN = np.linspace(0.0, 3.0, 61)   # mu_B / T values for the observables vs mu

here = Path(__file__).parent


def jk_error(est):
    """Jackknife error; the sample axis is axis 0."""
    return np.sqrt((N_JK - 1) / N_JK * ((est - est.mean(axis=0)) ** 2).sum(axis=0))


def trapz(y, x):
    """Trapezoid integral of y over the last axis."""
    return (0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x)).sum(axis=-1)


def log_trapz_exp(lr, x):
    """log of int exp(lr) dx over the last axis, without overflow."""
    top = lr.max(axis=-1, keepdims=True)
    return top[..., 0] + np.log(trapz(np.exp(lr - top), x))


# ---- raw data --------------------------------------------------------------
def load_raw(name):
    d = np.loadtxt(here / "raw" / f"{name}.dat")
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3:3 + N_JK]


data = {}
c_raw, F, F_err, F_jk = load_raw("R")
data["dlogrho"] = (P * (F - c_raw), P * F_err, P * (F_jk - c_raw[:, None]))
for n in ORDERS:
    c_n, mean, err, jk = load_raw(f"B{n}")
    assert np.allclose(c_n, c_raw), f"c values of B{n}.dat do not match R.dat"
    data[f"B{n}"] = (mean, err, jk)


def in_range(lo, hi):
    return (c_raw >= lo - 1e-9) & (c_raw <= hi + 1e-9)


def chi2(name, lo, hi, deg):
    mean, err, _ = data[name]
    m = in_range(lo, hi)
    coef = poly.polyfit(c_raw[m], mean[m], deg, w=1 / err[m])
    return (((mean[m] - poly.polyval(c_raw[m], coef)) / err[m]) ** 2).sum(), m.sum() - (deg + 1)


def aic(name, lo, hi, deg):
    n_cut = len(c_raw) - in_range(lo, hi).sum()
    return chi2(name, lo, hi, deg)[0] + 2 * (deg + 1) + 2 * n_cut


def best_degree(name, lo, hi):
    return min(range(1, MAX_DEGREE + 1), key=lambda d: aic(name, lo, hi, d))


def fit_on_grid(name, lo, hi, deg, grid):
    """Fit mean and every jackknife sample; return values on grid, shape (49, n_grid)."""
    mean, err, jk = data[name]
    m = in_range(lo, hi)
    # polyfit fits every column of a 2D y separately, with the same weights
    y = np.column_stack([mean[m], jk[m]])
    coef = poly.polyfit(c_raw[m], y, deg, w=1 / err[m])  # shape (deg+1, 49)
    return poly.polyval(grid, coef)                       # shape (49, n_grid)


# ---- rho(c, mu) and observables for one fit variant -------------------------
def log_rho_unnorm(fits, grid, mu, n_max=8):
    """log rho_0(c) + V sum_n B_n mu^n/n!  (not normalised), shape (49, n_grid)."""
    steps = 0.5 * (fits["dlogrho"][:, 1:] + fits["dlogrho"][:, :-1]) * np.diff(grid)
    lr = np.concatenate([np.zeros((steps.shape[0], 1)), np.cumsum(steps, axis=1)], axis=1)
    return lr + V * sum(fits[f"B{n}"] * mu ** n / factorial(n) for n in ORDERS if n <= n_max)


def log_rho_mu(fits, grid, mu, n_max=8):
    """Normalised log rho(c, mu): int rho dc = 1."""
    lr = log_rho_unnorm(fits, grid, mu, n_max)
    return lr - log_trapz_exp(lr, grid)[:, None]


def observables(fits, grid, mu, n_max=8):
    """<c>, width, Delta p / T^4, n_B / T^3, each shape (49,)."""
    rho = np.exp(log_rho_mu(fits, grid, mu, n_max))
    avg = trapz(rho * grid, grid)
    width = np.sqrt(trapz(rho * grid ** 2, grid) - avg ** 2)
    log_z_ratio = (log_trapz_exp(log_rho_unnorm(fits, grid, mu, n_max), grid)
                   - log_trapz_exp(log_rho_unnorm(fits, grid, 0.0, n_max), grid))
    dexp = sum(fits[f"B{n}"] * mu ** (n - 1) / factorial(n - 1) for n in ORDERS if n <= n_max)
    density = trapz(rho * dexp, grid)
    return avg, width, log_z_ratio / V, density


OBS = ["mean_c", "width", "pressure", "density"]

# the rho(c, mu) curves are compared on the narrowest grid
grid0 = np.round(np.arange(RANGES[0][0], RANGES[0][1] + DC / 2, DC), 6)

variants = []
for lo, hi in RANGES:
    grid = np.round(np.arange(lo, hi + DC / 2, DC), 6)
    on_grid0 = np.isin(grid, grid0)
    best = {name: best_degree(name, lo, hi) for name in data}
    for off in DEGREE_OFFSETS:
        degs = {name: max(1, best[name] + off) for name in data}
        fits = {name: fit_on_grid(name, lo, hi, degs[name], grid) for name in data}
        v = {"label": f"[{lo}, {hi}] {off:+d}", "range": (lo, hi), "fits": fits, "grid": grid,
             "degs": degs, "aic": sum(aic(name, lo, hi, degs[name]) for name in data)}
        # all arrays below have the jackknife axis (49) first
        v["log_rho"] = np.stack([log_rho_mu(fits, grid, mu)[:, on_grid0] for mu in MU_PLOT], axis=1)
        for n_max in (8, 6):
            obs = [observables(fits, grid, mu, n_max) for mu in MU_SCAN]
            for k, key in enumerate(OBS):
                v[f"{key}{n_max}"] = np.stack([o[k] for o in obs], axis=1)  # (49, n_mu_scan)
        variants.append(v)

aics = np.array([v["aic"] for v in variants])
weights = np.exp(-(aics - aics.min()) / 2)
weights /= weights.sum()

print("fit variants: range, degree offset, degrees, AIC, weight")
for v, w in zip(variants, weights):
    degs = " ".join(f"{name}:{d}" for name, d in v["degs"].items())
    print(f"  {v['label']:>16}  {degs}  AIC = {v['aic'] - aics.min():7.2f}  w = {w:.3f}")


def combine(key, transform=lambda x: x):
    """Model-averaged central value, stat, sys and total error of variants[*][key]."""
    vals = np.array([transform(v[key]) for v in variants])  # (n_var, 49, ...)
    avg = np.tensordot(weights, vals, axes=1)               # (49, ...)
    central = avg[0]
    stat = jk_error(avg[1:])
    sys = np.sqrt(np.tensordot(weights, (vals[:, 0] - central) ** 2, axes=1))
    return central, stat, sys, np.sqrt(stat ** 2 + sys ** 2)


# ---- output: rho(c, mu) -----------------------------------------------------
lr, lr_stat, lr_sys, lr_tot = combine("log_rho")
rho, rho_stat, rho_sys, rho_tot = combine("log_rho", np.exp)

cols, header = [grid0], "c"
for i, mu in enumerate(MU_PLOT):
    cols += [lr[i], lr_stat[i], lr_sys[i], rho[i], rho_stat[i], rho_sys[i]]
    header += f"  log_rho(mu={mu}) stat sys  rho(mu={mu}) stat sys"
np.savetxt(here / "rho_mu.dat", np.column_stack(cols), header=header)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
for i, mu in enumerate(MU_PLOT):
    line, = ax1.plot(grid0, rho[i], label=rf"$\mu_B/T = {mu}$")
    ax1.fill_between(grid0, rho[i] - rho_tot[i], rho[i] + rho_tot[i], color=line.get_color(), alpha=0.3)
    ax2.plot(grid0, lr[i], color=line.get_color())
    ax2.fill_between(grid0, lr[i] - lr_tot[i], lr[i] + lr_tot[i], color=line.get_color(), alpha=0.3)
ax1.set_ylabel(r"$\rho(c, \mu_B)$")
ax1.legend()
ax2.set_ylabel(r"$\log \rho(c, \mu_B)$")
ax2.set_xlabel("c")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
ax1.set_title("bands: stat + sys (AIC model average) in quadrature", fontsize=9)
fig.tight_layout()
fig.savefig(here / "rho_mu.png", dpi=150)

# ---- output: observables vs mu ---------------------------------------------
labels = {"mean_c": r"$\langle c \rangle_{\mu_B}$",
          "width": r"$\sqrt{\langle c^2 \rangle - \langle c \rangle^2}$",
          "pressure": r"$\Delta p / T^4$",
          "density": r"$n_B / T^3$"}
fig2, axes2 = plt.subplots(len(OBS), 1, figsize=(7, 2.6 * len(OBS)), sharex=True)
cols, header = [MU_SCAN], "mu_B/T"
for n_max, color in [(8, "C0"), (6, "C1")]:
    for ax, key in zip(axes2, OBS):
        val, stat, sys, tot = combine(f"{key}{n_max}")
        cols += [val, stat, sys]
        header += f"  {key}(n<={n_max}) stat sys"
        ax.plot(MU_SCAN, val, color=color, ls="-" if n_max == 8 else "--", label=rf"$n \leq {n_max}$")
        ax.fill_between(MU_SCAN, val - tot, val + tot, color=color, alpha=0.3)
np.savetxt(here / "thermo_mu.dat", np.column_stack(cols), header=header)
for ax, key in zip(axes2, OBS):
    ax.set_ylabel(labels[key])
    ax.grid(alpha=0.3)
axes2[0].legend()
axes2[-1].set_xlabel(r"$\mu_B / T$")
fig2.tight_layout()
fig2.savefig(here / "thermo_mu.png", dpi=150)

# ---- check: fits against the raw data, one band per fit range ---------------
fig3, axes3 = plt.subplots(len(data), 1, figsize=(7, 2.6 * len(data)), sharex=True)
shown = [max((i for i, v in enumerate(variants) if v["range"] == r), key=lambda i: weights[i])
         for r in RANGES]  # most probable variant of each range
for ax, name in zip(axes3, data):
    mean, err, _ = data[name]
    ax.errorbar(c_raw, mean, yerr=err, fmt="o", ms=2, capsize=1.5, color="k", label="raw data")
    for i, color in zip(shown, ["C3", "C0"]):
        v = variants[i]
        f = v["fits"][name]
        f_err = jk_error(f[1:])
        ax.plot(v["grid"], f[0], color=color, label=f"{v['label']} (w = {weights[i]:.2f})")
        ax.fill_between(v["grid"], f[0] - f_err, f[0] + f_err, color=color, alpha=0.3)
    ax.set_ylabel(r"$d\log\rho_0/dc$" if name == "dlogrho" else name)
    ax.grid(alpha=0.3)
axes3[0].legend(fontsize=8)
axes3[-1].set_xlabel("c")
fig3.tight_layout()
fig3.savefig(here / "fit_raw.png", dpi=150)
plt.show()
