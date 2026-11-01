"""Baryon-number susceptibilities from the density of states (DoS_Notes-1.pdf, Eq. 10).

    chi^B_m(mu) = (1/V) d^m/dmu^m  log int dc rho_0(c) exp( G(c, mu) ),
    G(c, mu)    = V sum_{n=2, even}^{N_max} B_n(c) mu^n / n!,     B_n = chi^B_n(c)

The derivatives are done exactly, not by finite differences:
  * with g_k(c) = d^k G / dmu^k = V sum_n B_n(c) mu^(n-k) / (n-k)!,
    d^m exp(G) / dmu^m = exp(G) * Y_m(g_1, ..., g_m)   (complete Bell polynomial)
  * so the raw moments of the "mu-derivative" are M_m = < Y_m >_mu,
    averaged with the normalised rho(c, mu),
  * and d^m log Z / dmu^m = kappa_m, the cumulants belonging to these moments:
    kappa_m = M_m - sum_{k=1}^{m-1} binom(m-1, k-1) kappa_k M_{m-k}.
  chi^B_m = kappa_m / V.
At mu = 0 this gives e.g. chi_2 = <B_2>_0, chi_4 = <B_4>_0 + 3 V (<B_2^2>_0 - <B_2>_0^2),
which is Eq. (21) of the notes with <chi_1> = 0.

Data, fits, jackknife and AIC model averaging: see dos_common.py.
"""
from math import comb, factorial

import numpy as np
import matplotlib.pyplot as plt

from dos_common import (V, ORDERS, here, trapz, log_trapz_exp, model_average,
                        build_variants, print_variants, log_rho0, exponent)

N_MAX_LIST = [8, 6]                   # truncations of the sum over n, for comparison
M_MAX = 8                             # compute chi_1 ... chi_M_MAX
MU_SCAN = np.linspace(0.0, 3.0, 61)   # mu_B / T values


def susceptibilities(fits, grid, mu, n_max):
    """chi_1 ... chi_M_MAX at this mu, shape (M_MAX, 49)."""
    orders = [n for n in ORDERS if n <= n_max]

    lr = log_rho0(fits, grid) + exponent(fits, mu, n_max)
    rho = np.exp(lr - log_trapz_exp(lr, grid)[:, None])  # normalised rho(c, mu)

    # g_k = d^k G / dmu^k, k = 1 .. M_MAX  (g[0] is unused)
    zero = np.zeros_like(rho)
    g = [zero] + [V * sum((fits[f"B{n}"] * mu ** (n - k) / factorial(n - k) for n in orders if n >= k),
                          zero) for k in range(1, M_MAX + 1)]

    # complete Bell polynomials: Y_0 = 1, Y_{m+1} = sum_k binom(m, k) Y_{m-k} g_{k+1}
    Y = [np.ones_like(rho)]
    for m in range(M_MAX):
        Y.append(sum(comb(m, k) * Y[m - k] * g[k + 1] for k in range(m + 1)))

    # moments -> cumulants
    M = [None] + [trapz(rho * Y[m], grid) for m in range(1, M_MAX + 1)]
    kappa = [None]
    for m in range(1, M_MAX + 1):
        kappa.append(M[m] - sum(comb(m - 1, k - 1) * kappa[k] * M[m - k] for k in range(1, m)))
    return np.array(kappa[1:]) / V


variants, weights = build_variants()
print_variants(variants, weights)

# vals[n_max]: (n_variants, 49, M_MAX, n_mu)
vals = {n_max: [] for n_max in N_MAX_LIST}
for v in variants:
    grid = v.grid()
    fits = v.fits(grid)
    for n_max in N_MAX_LIST:
        chi = np.stack([susceptibilities(fits, grid, mu, n_max) for mu in MU_SCAN], axis=2)
        vals[n_max].append(chi.transpose(1, 0, 2))
results = {n_max: model_average(np.array(vals[n_max]), weights)[:3] for n_max in N_MAX_LIST}

# ---- table at mu = 0 --------------------------------------------------------
print("chi^B_m at mu_B = 0   (central +- stat +- sys)")
for n_max in N_MAX_LIST:
    val, stat, sys = results[n_max]
    print(f"  N_max = {n_max}:")
    for m in range(2, M_MAX + 1, 2):
        print(f"    chi_{m} = {val[m - 1, 0]: .5e} +- {stat[m - 1, 0]:.1e} +- {sys[m - 1, 0]:.1e}")

# ---- save and plot vs mu ----------------------------------------------------
cols, header = [MU_SCAN], "mu_B/T"
for n_max in N_MAX_LIST:
    val, stat, sys = results[n_max]
    for m in range(1, M_MAX + 1):
        cols += [val[m - 1], stat[m - 1], sys[m - 1]]
        header += f"  chi{m}(Nmax={n_max}) stat sys"
np.savetxt(here / "chi_calc.dat", np.column_stack(cols), header=header)

fig, axes = plt.subplots(M_MAX // 2, 2, figsize=(10, 2.4 * M_MAX // 2), sharex=True)
for m, ax in zip(range(1, M_MAX + 1), axes.flat):
    for n_max, color, ls in [(8, "C0", "-"), (6, "C1", "--")]:
        val, stat, sys = results[n_max]
        tot = np.sqrt(stat[m - 1] ** 2 + sys[m - 1] ** 2)
        ax.plot(MU_SCAN, val[m - 1], color=color, ls=ls, label=rf"$N_{{max}} = {n_max}$")
        ax.fill_between(MU_SCAN, val[m - 1] - tot, val[m - 1] + tot, color=color, alpha=0.3)
    ax.set_ylabel(rf"$\chi^B_{m}$")
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.3)
axes.flat[0].legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel(r"$\mu_B / T$")
fig.suptitle("bands: stat + sys (AIC model average) in quadrature", fontsize=9)
fig.tight_layout()
fig.savefig(here / "chi_calc.png", dpi=150)
plt.show()
