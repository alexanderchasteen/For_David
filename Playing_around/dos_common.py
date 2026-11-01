"""Shared pieces of the density-of-states analysis (used by chi_calc.py, lee_yang.py, ly_tscan.py).

Conventions (DoS_Notes-1.pdf, Eqs. 2, 4, 8-10)
    constrained action  S_c = S + (P/2) (F - c)^2,       P = 2e4
    d log rho_0 / dc    = P ( <F>_c - c )
    rho(c, mu)          = rho_0(c) exp( V sum_{n even} B_n(c) mu^n / n! ),  B_n = chi^B_n(c)
    mu = mu_B / T,  V = V T^3 = (N_s / N_t)^3 = (16 / 8)^3 = 8.

Input: R.dat (= <F>_c) and B{n}.dat from katinka:/data/4hex/ana_dos/16x8/<T>, copied to
raw/<T>/. Columns: c, mean, error, 48 jackknife samples. Each c is a separate simulation.
The data directory is raw/150 unless the environment variable DOS_RAW says otherwise;
configure(dir) switches to another one.

Fits: polynomial in c, weighted by 1/error, done on the mean and on every jackknife
sample (same weights). Several variants (fit range x degree) are combined by AIC
model averaging, AIC = chi^2 + 2 n_params + 2 n_cut (n_cut = points outside the range).
Fit ranges: all points, and all but the 3 lowest and 3 highest c values.

Array convention: axis 0 of every per-sample array has length 49:
index 0 is the fit to the means (central value), indices 1..48 the jackknife samples.
"""
import os
from math import factorial
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as poly

P = 2e4
V = 8.0
N_JK = 48
ORDERS = [2, 4, 6, 8]
DEGREE_OFFSETS = [-1, 0, 1, 2, 3]     # degrees tried: AIC-best + these
DC = 0.005                            # default c spacing of the integration grid
TRIM = 3                              # points dropped at each end for the narrower fit range

here = Path(__file__).parent


# ---- small numerics ---------------------------------------------------------
def jk_error(est):
    """Jackknife error of est over axis 0 (the 48 samples, without the central value)."""
    return np.sqrt((N_JK - 1) / N_JK * ((est - est.mean(axis=0)) ** 2).sum(axis=0))


def trapz(y, x):
    """Trapezoid integral of y over the last axis."""
    return (0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x)).sum(axis=-1)


def log_trapz_exp(lr, x):
    """log of int exp(lr) dx over the last axis, without overflow. lr may be complex."""
    top = lr.real.max(axis=-1, keepdims=True)
    return top[..., 0] + np.log(trapz(np.exp(lr - top), x))


def model_average(vals, weights):
    """AIC model average of vals (n_variants, 49, ...): central, stat, sys, total."""
    avg = np.tensordot(weights, vals, axes=1)  # (49, ...)
    central = avg[0]
    stat = jk_error(avg[1:])
    sys = np.sqrt(np.tensordot(weights, (vals[:, 0] - central) ** 2, axes=1))
    return central, stat, sys, np.sqrt(stat ** 2 + sys ** 2)


# ---- raw data ---------------------------------------------------------------
RAW_DIR = None
c_raw = None
data = {}
NAMES = ["dlogrho"] + [f"B{n}" for n in ORDERS]
RANGES = []
MAX_DEGREE = 10


def configure(raw_dir):
    """Load R.dat and B{n}.dat from raw_dir and set the fit ranges for this data set.

    Only c values present in every file are kept (the k-th occurrence of a repeated
    c value in one file is matched with the k-th occurrence in the others)."""
    global RAW_DIR, c_raw, data, RANGES, MAX_DEGREE
    RAW_DIR = Path(raw_dir)
    raw = {name: np.loadtxt(RAW_DIR / f"{name}.dat", ndmin=2) for name in ["R"] + [f"B{n}" for n in ORDERS]}

    def keys(d):
        out, seen = [], {}
        for c in np.round(d[:, 0], 6):
            seen[c] = seen.get(c, 0) + 1
            out.append((c, seen[c]))
        return out

    k = {name: keys(d) for name, d in raw.items()}
    common = [key for key in k["R"] if all(key in k[name] for name in raw)]
    common.sort()
    rows = {name: np.array([raw[name][k[name].index(key)] for key in common]) for name in raw}

    c_raw = rows["R"][:, 0]
    F, F_err, F_jk = rows["R"][:, 1], rows["R"][:, 2], rows["R"][:, 3:3 + N_JK]
    data = {"dlogrho": (P * (F - c_raw), P * F_err, P * (F_jk - c_raw[:, None]))}
    for n in ORDERS:
        d = rows[f"B{n}"]
        data[f"B{n}"] = (d[:, 1], d[:, 2], d[:, 3:3 + N_JK])

    n_pts = len(c_raw)
    RANGES = [(c_raw[0], c_raw[-1])]
    if n_pts >= 20:
        RANGES.append((c_raw[TRIM], c_raw[-1 - TRIM]))
    MAX_DEGREE = max(2, min(10, n_pts // 4))


configure(os.environ.get("DOS_RAW", here / "raw" / "150"))


# ---- polynomial fits --------------------------------------------------------
def in_range(lo, hi):
    return (c_raw >= lo - 1e-9) & (c_raw <= hi + 1e-9)


def fit_coef(name, lo, hi, deg):
    """Polynomial coefficients for the mean and each sample, shape (deg+1, 49)."""
    mean, err, jk = data[name]
    m = in_range(lo, hi)
    y = np.column_stack([mean[m], jk[m]])
    return poly.polyfit(c_raw[m], y, deg, w=1 / err[m])


def chi2(name, lo, hi, deg):
    mean, err, _ = data[name]
    m = in_range(lo, hi)
    fit = poly.polyval(c_raw[m], fit_coef(name, lo, hi, deg)[:, 0])
    return (((mean[m] - fit) / err[m]) ** 2).sum(), m.sum() - (deg + 1)


def aic(name, lo, hi, deg):
    n_cut = len(c_raw) - in_range(lo, hi).sum()
    return chi2(name, lo, hi, deg)[0] + 2 * (deg + 1) + 2 * n_cut


def best_degree(name, lo, hi):
    return min(range(1, MAX_DEGREE + 1), key=lambda d: aic(name, lo, hi, d))


class Variant:
    """One fit variant: a c range and a polynomial degree for every fitted quantity."""

    def __init__(self, lo, hi, offset):
        self.range = (lo, hi)
        self.offset = offset
        self.label = f"[{lo:g}, {hi:g}] {offset:+d}"
        n_in = in_range(lo, hi).sum()
        self.degs = {name: int(np.clip(best_degree(name, lo, hi) + offset, 1, n_in - 2)) for name in NAMES}
        self.coef = {name: fit_coef(name, lo, hi, self.degs[name]) for name in NAMES}
        self.aic = sum(aic(name, lo, hi, self.degs[name]) for name in NAMES)

    def grid(self, dc=DC):
        lo, hi = self.range
        return np.round(np.arange(lo, hi + dc / 2, dc), 9)

    def fits(self, grid):
        """Fitted curves on grid: {name: array (49, n_grid)}."""
        return {name: poly.polyval(grid, self.coef[name]) for name in NAMES}


def build_variants():
    """All fit variants and their AIC weights."""
    variants = [Variant(lo, hi, off) for lo, hi in RANGES for off in DEGREE_OFFSETS]
    aics = np.array([v.aic for v in variants])
    weights = np.exp(-(aics - aics.min()) / 2)
    return variants, weights / weights.sum()


def print_variants(variants, weights):
    print(f"data: {RAW_DIR}  ({len(c_raw)} c values from {c_raw[0]:g} to {c_raw[-1]:g})")
    print("fit variants: range, degree offset, degrees, AIC - AIC_min, weight")
    amin = min(v.aic for v in variants)
    for v, w in zip(variants, weights):
        degs = " ".join(f"{name}:{d}" for name, d in v.degs.items())
        print(f"  {v.label:>18}  {degs}  AIC = {v.aic - amin:7.2f}  w = {w:.3f}")


# ---- density of states ------------------------------------------------------
def log_rho0(fits, grid):
    """Unnormalised log rho_0(c) on grid by trapezoid integration, shape (49, n_grid)."""
    d = fits["dlogrho"]
    steps = 0.5 * (d[:, 1:] + d[:, :-1]) * np.diff(grid)
    return np.concatenate([np.zeros((d.shape[0], 1)), np.cumsum(steps, axis=1)], axis=1)


SCHEMES = ("taylor", "fugacity")


def fugacity_coefs(fits, n_max=8):
    """Sector coefficients p_B(c), B = 1 .. n_max/2, of the constrained fugacity expansion

        Delta p(c) = sum_B p_B(c) (cosh(B mu) - 1)          (notes, Eq. 25-26)

    fixed by matching its Taylor coefficients to chi_2 .. chi_{n_max}:  sum_B p_B B^n = chi_n.
    Returns an array (n_sectors, ...) with the same trailing shape as the fits."""
    orders = [n for n in ORDERS if n <= n_max]
    sectors = np.arange(1, len(orders) + 1)
    M = np.array([[float(B) ** n for B in sectors] for n in orders])
    chi = np.stack([fits[f"B{n}"] for n in orders])
    return np.tensordot(np.linalg.inv(M), chi, axes=1)


def exponent(fits, mu, n_max=8, scheme="taylor", b_max=None):
    """The exponent G(c, mu) of rho(c, mu) = rho_0 exp(G), for mu scalar or broadcastable.

    taylor:    V sum_n B_n(c) mu^n / n!                      (notes, Eq. 8)
    fugacity:  V sum_{B <= b_max} p_B(c) (cosh(B mu) - 1)    (notes, Eq. 25-28), 2 pi-periodic in Im mu;
               the p_B are matched at order n_max (n_max/2 sectors) and the sum is then cut at
               b_max sectors, as in the notes' Fig. 3 (b_max = None keeps all of them)
    Both agree to order mu^n_max when all sectors are kept."""
    if scheme == "taylor":
        return V * sum(fits[f"B{n}"] * mu ** n / factorial(n) for n in ORDERS if n <= n_max)
    p = fugacity_coefs(fits, n_max)[:b_max]
    return V * sum(p[k] * (np.cosh((k + 1) * mu) - 1) for k in range(len(p)))


def dexponent(fits, mu, n_max=8, scheme="taylor", b_max=None):
    """dG/dmu for the same schemes."""
    if scheme == "taylor":
        return V * sum(fits[f"B{n}"] * mu ** (n - 1) / factorial(n - 1) for n in ORDERS if n <= n_max)
    p = fugacity_coefs(fits, n_max)[:b_max]
    return V * sum(p[k] * (k + 1) * np.sinh((k + 1) * mu) for k in range(len(p)))


def coverage(variant, dc=DC):
    """How well the c range covers rho_0: (fraction of rho_0 in the outer 5% of the grid,
    position of the peak as a fraction of the range). Used to reject sparse data sets."""
    grid = variant.grid(dc)
    lr = log_rho0(variant.fits(grid), grid)[0]
    r = np.exp(lr - lr.max())
    k = max(1, int(0.05 * len(grid)))
    edge = (trapz(r[:k], grid[:k]) + trapz(r[-k:], grid[-k:])) / trapz(r, grid)
    return edge, (grid[np.argmax(r)] - grid[0]) / (grid[-1] - grid[0])
