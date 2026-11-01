"""Lee-Yang zeros of the DoS-resummed partition function ratio (DoS_Notes-1.pdf, Eq. 9).

    Z(mu)/Z(0) = int dc rho_0(c) exp(G(c, mu)) / int dc rho_0(c),
    G(c, mu)   = V sum_{n=2, even}^{N_max} B_n(c) mu^n / n!,   mu = mu_B / T (complex)

Z is an entire function of mu; its zeros are the Lee-Yang zeros. Since the B_n are
real and n is even, Z(-mu) = Z(mu) and Z(conj mu) = conj Z(mu): zeros come in
quadruplets +-mu, +-conj(mu), and it is enough to search Re mu > 0, Im mu > 0.
On both axes G is real, so the integrand is positive and Z cannot vanish there.

Method
  1. Scan: log Z(mu)/Z(0) on a fine grid in the first quadrant (central fit of the
     most probable variant). A cell contains a zero when the phase of Z winds by
     +2 pi around it (argument principle).
  2. Refine: Newton iteration  mu <- mu - Z/Z' = mu - 1 / <g_1>_mu,  with
     g_1 = dG/dmu and <.>_mu the complex-weighted average with weight rho_0 exp(G).
  3. Cheap checks on every zero:
       edge      fraction of |integrand| in the outer 5% of the c grid; a zero produced
                 by the ends of the fit range is an artifact of the extrapolated fits
       d_other   distance to the nearest zero of the other truncation (N_max 8 <-> 6);
                 zeros that move with N_max are expansion artifacts (notes, Sec. 2.3.1)
       isolation distance to the nearest zero that failed one of the two tests above;
                 artifact zeros come in dense rows along the red blobs of the heat map,
                 physical candidates sit away from them
     Zeros passing all three are candidates.
  4. On the candidates: winding number on a small circle (confirms a simple zero),
     shift when the c spacing is halved (integration error), and the errors:
     Newton is re-run from the central zero on every jackknife sample of every fit
     variant: stat = jackknife, sys = AIC-weighted spread over variants. Samples whose
     Newton run ends at a different zero are counted as outliers and left out.
     A candidate is "stable" if all of these checks pass too.
  Following the notes, the LY-zero candidate reported is the stable zero with the
  smallest Im mu; the one with the smallest |mu| sets the radius of convergence of
  the Taylor series of log Z around mu = 0.

Data, fits, jackknife and AIC model averaging: see dos_common.py.
"""
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dos_common import (V, DC, N_JK, SCHEMES, here, jk_error, trapz, log_trapz_exp,
                        build_variants, print_variants, log_rho0, exponent, dexponent)

T_MEV = float(os.environ.get("DOS_T", 150))   # temperature (folder name on katinka); only for the MeV values
OUT = Path(os.environ.get("DOS_OUT", here))     # where lee_yang.dat / .png are written
SCHEME = os.environ.get("DOS_SCHEME", "taylor")  # inner expansion: "taylor" (Eq. 8) or "fugacity" (Eq. 25)
assert SCHEME in SCHEMES, SCHEME
TAG = "" if SCHEME == "taylor" else f"_{SCHEME}"  # suffix of the output files
MAX_CELLS = 2500            # at most this many scan cells are refined (the ones closest to the origin)
N_MAX_LIST = [8, 6]                 # taylor: the two truncation orders compared
FUGACITY_TRUNCS = [(8, 2), (8, 1)]  # fugacity: (matching order, sectors kept) pairs compared, as in the notes' Fig. 3
RE_MAX, IM_MAX = 9.0, 9.0   # search box in the complex mu_B/T plane (low-T zeros sit at |mu| ~ 6-7)
SCAN_STEP = 0.01            # grid for the zero search
MAP_STEP = 0.02             # grid for the plotted heat map
EDGE_FRAC = 0.05            # outer fraction of the c grid used for the edge check
EDGE_TOL = 1e-3             # max tolerated |integrand| fraction at the edges
OTHER_TOL = 0.6             # max distance to a zero of the other truncation (the distance itself is reported)
ISO_TOL = 0.15              # min distance to the nearest artifact zero of the same truncation
MATCH_TOL = 0.3             # a sample/variant zero further than this from the central one is an outlier
GRID_TOL = 1e-3             # max shift when the c spacing is halved
MIN_WEIGHT = 0.5            # min total AIC weight of the variants that have the zero
MAX_OUTLIER = 0.1           # max fraction of jackknife samples that are outliers
MAX_CANDIDATES = 10         # the full checks run only on this many candidates (smallest |mu| first)
IM_WINDOW = 1.5             # the smallest-Im candidate is chosen among stable zeros with |mu| <= IM_WINDOW * min |mu|
if SCHEME == "fugacity":
    N_MAX_LIST = FUGACITY_TRUNCS
    # cosh(B mu) is 2 pi-periodic in Im mu and Z(-conj(mu) + 2 pi i) = conj Z(mu) (Roberge-Weiss
    # mirror at Im mu = pi), so Im mu in (0, pi] is the whole fundamental domain; beyond Re mu ~ 6
    # the highest sector e^(B_max Re mu) dominates everything.
    RE_MAX, IM_MAX = 6.0, float(np.pi)


def targs(tr):
    """Keyword arguments of exponent()/dexponent() for a truncation label."""
    return dict(n_max=tr, scheme=SCHEME) if SCHEME == "taylor" else dict(n_max=tr[0], b_max=tr[1], scheme=SCHEME)


def tcode(tr):
    """Integer stored in the first column of lee_yang.dat: N_max (taylor) or the number of sectors (fugacity)."""
    return tr if SCHEME == "taylor" else tr[1]


def tname(tr):
    return f"N_max = {tr}" if SCHEME == "taylor" else f"{tr[1]} sector{'s' if tr[1] > 1 else ''} (matched at N_max = {tr[0]})"


def log_z_ratio(fits0, lr0, grid, mus, n_max, chunk=2000):
    """log Z(mu)/Z(0) for the central fit (fits0 arrays are 1-D) at an array of complex mu."""
    log_z0 = log_trapz_exp(lr0, grid)
    fits0_row = {k: f[None, :] for k, f in fits0.items()}
    flat = mus.ravel()
    out = np.empty(flat.shape, complex)
    for i in range(0, flat.size, chunk):
        mu = flat[i:i + chunk, None]
        lr = lr0[None, :] + exponent(fits0_row, mu, **targs(n_max))
        out[i:i + chunk] = log_trapz_exp(lr, grid) - log_z0
    return out.reshape(mus.shape)


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def winding_cells(phase):
    """Winding number of the phase around every grid cell; rows = Im, columns = Re."""
    d = (wrap(phase[:-1, 1:] - phase[:-1, :-1]) + wrap(phase[1:, 1:] - phase[:-1, 1:])
         + wrap(phase[1:, :-1] - phase[1:, 1:]) + wrap(phase[:-1, :-1] - phase[1:, :-1]))
    return np.rint(d / (2 * np.pi)).astype(int)


def newton(fits, lr0, grid, mu0, n_max, max_iter=100, tol=1e-13, max_step=0.2):
    """Newton zero search from mu0 for all 49 samples at once.

    Returns mu (49,), converged (49,), and the relative residual |int w| / int |w| (49,)."""
    mu = np.full(lr0.shape[0], mu0, complex)
    done = np.zeros(mu.shape, bool)
    for _ in range(max_iter):
        lr = lr0 + exponent(fits, mu[:, None], **targs(n_max))
        w = np.exp(lr - lr.real.max(axis=1, keepdims=True))
        z = trapz(w, grid)
        step = -z / trapz(w * dexponent(fits, mu[:, None], **targs(n_max)), grid)   # -Z/Z' = -1/<G'>
        big = np.abs(step) > max_step
        step[big] *= max_step / np.abs(step[big])
        step[done] = 0
        mu = mu + step
        done |= np.abs(step) < tol * np.maximum(1, np.abs(mu))
        if done.all():
            break
    lr = lr0 + exponent(fits, mu[:, None], **targs(n_max))
    w = np.exp(lr - lr.real.max(axis=1, keepdims=True))
    residual = np.abs(trapz(w, grid)) / trapz(np.abs(w), grid)
    return mu, done & (residual < 1e-8), residual


def edge_fraction(fits0, lr0, grid, mu, n_max):
    """Fraction of int |rho_0 exp(G)| dc that comes from the outer EDGE_FRAC of the grid."""
    lr = lr0 + exponent(fits0, mu, **targs(n_max))
    a = np.abs(np.exp(lr - lr.real.max()))
    k = max(1, int(EDGE_FRAC * len(grid)))
    return (trapz(a[:k], grid[:k]) + trapz(a[-k:], grid[-k:])) / trapz(a, grid)


def circle_winding(fits0, lr0, grid, mu, n_max, radius=None, n_pts=64):
    radius = radius or 1e-3 * max(1.0, abs(mu))
    ring = mu + radius * np.exp(2j * np.pi * np.arange(n_pts + 1) / n_pts)
    phase = log_z_ratio(fits0, lr0, grid, ring, n_max).imag
    return int(np.rint(wrap(np.diff(phase)).sum() / (2 * np.pi)))


# ---- fits -------------------------------------------------------------------
variants, weights = build_variants()
print_variants(variants, weights)
ibase = int(np.argmax(weights))
base = variants[ibase]
grid_b = base.grid()
fits_b = base.fits(grid_b)
lr0_b = log_rho0(fits_b, grid_b)
fits0_b = {k: f[0] for k, f in fits_b.items()}
fits_c = {k: f[:1] for k, f in fits_b.items()}   # central fit alone, for the scan refinement
lr0_c = lr0_b[:1]
print(f"scheme: {SCHEME}; zero search on the central fit of variant {base.label} (weight {weights[ibase]:.3f})")

grid_f = base.grid(DC / 2)                   # finer grid for the integration check
fits_f = base.fits(grid_f)
lr0_f = log_rho0(fits_f, grid_f)

var_data = []                                # per-variant data for the error analysis
for v in variants:
    g = v.grid()
    f = v.fits(g)
    var_data.append((f, log_rho0(f, g), g))

# ---- 1. scan + 2. refine ----------------------------------------------------
re = np.arange(SCAN_STEP, RE_MAX + SCAN_STEP / 2, SCAN_STEP)
im = np.arange(SCAN_STEP, IM_MAX + SCAN_STEP / 2, SCAN_STEP)
mus = re[None, :] + 1j * im[:, None]

zeros = {}
for n_max in N_MAX_LIST:
    phase = log_z_ratio(fits0_b, lr0_b[0], grid_b, mus, n_max).imag
    wn = winding_cells(phase)
    cells = np.argwhere(wn != 0)
    n_cells = len(cells)
    if n_cells > MAX_CELLS:
        order = np.argsort(np.abs(re[cells[:, 1]] + 1j * im[cells[:, 0]]))
        cells = cells[order[:MAX_CELLS]]
    found = []
    for iy, ix in cells:
        mu0 = re[ix] + SCAN_STEP / 2 + 1j * (im[iy] + SCAN_STEP / 2)
        mu, ok, res = newton(fits_c, lr0_c, grid_b, mu0, n_max)
        if not ok[0] or abs(mu[0] - mu0) > 3 * SCAN_STEP:
            continue
        if any(abs(mu[0] - z["mu"]) < 1e-6 for z in found):
            continue
        found.append({"mu": mu[0], "residual": res[0]})
    found.sort(key=lambda z: abs(z["mu"]))
    zeros[n_max] = found
    print(f"{tname(n_max)}: {n_cells} grid cells with non-zero winding number"
          f"{f' (only the {MAX_CELLS} closest to the origin refined)' if n_cells > MAX_CELLS else ''}, "
          f"{len(found)} distinct zeros after Newton refinement", flush=True)

# ---- 3. cheap checks --------------------------------------------------------
for n_max in N_MAX_LIST:
    other = N_MAX_LIST[1 - N_MAX_LIST.index(n_max)]
    mus_all = np.array([z["mu"] for z in zeros[n_max]])
    mus_oth = np.array([z["mu"] for z in zeros[other]])
    for z in zeros[n_max]:
        z["edge"] = edge_fraction(fits0_b, lr0_b[0], grid_b, z["mu"], n_max)
        z["d_other"] = np.abs(mus_oth - z["mu"]).min() if len(mus_oth) else np.inf
        z["artifact"] = z["edge"] >= EDGE_TOL or z["d_other"] >= OTHER_TOL
    mus_art = np.array([z["mu"] for z in zeros[n_max] if z["artifact"]])
    for z in zeros[n_max]:
        # distance to the nearest zero already known to be an artifact (a close pair of
        # genuine zeros is allowed; a zero sitting in a row of artifacts is not)
        z["isolation"] = np.abs(mus_art - z["mu"]).min() if len(mus_art) else np.inf
        z["candidate"] = not z["artifact"] and z["isolation"] > ISO_TOL
        z["stable"] = False

# ---- 4. full checks and errors on the candidates ----------------------------
for n_max in N_MAX_LIST:
    cands = sorted([z for z in zeros[n_max] if z["candidate"]], key=lambda z: abs(z["mu"]))
    for z in cands[MAX_CANDIDATES:]:
        z["candidate"] = False   # too far from the real axis to be of interest; left unchecked
        z["unchecked"] = True
    for z in cands[:MAX_CANDIDATES]:
        mu_c = z["mu"]
        z["winding"] = circle_winding(fits0_b, lr0_b[0], grid_b, mu_c, n_max)
        mu_f, ok_f, _ = newton(fits_f, lr0_f, grid_f, mu_c, n_max)
        z["d_grid"] = abs(mu_f[0] - mu_c) if ok_f[0] else np.inf

        per_var, w_used, n_out = [], [], 0
        for (f, lr0, g), w in zip(var_data, weights):
            mu, ok, _ = newton(f, lr0, g, mu_c, n_max)
            if not ok[0] or abs(mu[0] - mu_c) > MATCH_TOL:
                continue  # this variant has no matching zero
            bad = ~ok | (np.abs(mu - mu[0]) > MATCH_TOL)
            bad[0] = False
            # a sample whose Newton run wandered off is retried with smaller steps before
            # it is declared an outlier
            for max_step in (0.05, 0.02):
                if not bad.any():
                    break
                mu2, ok2, _ = newton(f, lr0, g, mu_c, n_max, max_step=max_step)
                fix = bad & ok2 & (np.abs(mu2 - mu[0]) <= MATCH_TOL)
                mu[fix] = mu2[fix]
                bad &= ~fix
            n_out += int(bad.sum())
            mu[bad] = mu[0]  # outliers (Newton ended at another zero) do not enter the error
            per_var.append(mu)
            w_used.append(w)
        z["weight"] = float(np.sum(w_used))
        z["n_variants"] = len(per_var)
        z["n_outliers"] = n_out
        if per_var:
            per_var = np.array(per_var)
            w_used = np.array(w_used) / np.sum(w_used)
            avg = np.tensordot(w_used, per_var, axes=1)
            z["central"] = avg[0]
            z["stat"] = jk_error(avg[1:].real) + 1j * jk_error(avg[1:].imag)
            dev = per_var[:, 0] - avg[0]
            z["sys"] = (np.sqrt(np.tensordot(w_used, dev.real ** 2, axes=1))
                        + 1j * np.sqrt(np.tensordot(w_used, dev.imag ** 2, axes=1)))
            z["stable"] = (z["winding"] == 1 and z["d_grid"] < GRID_TOL and z["weight"] >= MIN_WEIGHT
                           and n_out <= MAX_OUTLIER * N_JK * len(per_var))

# ---- report -----------------------------------------------------------------
rows = []
for n_max in N_MAX_LIST:
    zs = zeros[n_max]
    cands = [z for z in zs if z["candidate"]]
    arts = [z for z in zs if not z["candidate"]]
    n_unchecked = sum(z.get("unchecked", False) for z in zs)
    print(f"\n{tname(n_max)}: {len(zs)} zeros in the box, {len(cands)} candidates checked"
          f"{f' ({n_unchecked} more candidates further from the real axis left unchecked)' if n_unchecked else ''}, "
          f"{len(arts) - n_unchecked} artifacts (|mu| from {min(abs(z['mu']) for z in arts):.2f} to "
          f"{max(abs(z['mu']) for z in arts):.2f}; failed checks: "
          f"edge {sum(z['edge'] >= EDGE_TOL for z in arts)}, N_max-dependence {sum(z['d_other'] >= OTHER_TOL for z in arts)}, "
          f"isolation {sum(z['isolation'] <= ISO_TOL for z in arts)})")
    if cands:
        print(f"  candidates (mu = mu_B/T; central +- stat +- sys; T = {T_MEV:.0f} MeV for the MeV values)")
        print("     Re mu                      Im mu                      |mu|   mu_B [MeV]       d_other  isol   edge     d_grid   wind  weight  outliers  stable")
    for z in cands:
        if "central" not in z:
            print(f"    {z['mu'].real:6.3f}                     {z['mu'].imag:6.3f}                     {abs(z['mu']):5.3f}"
                  f"   -- no fit variant reproduces this zero (d_other {z['d_other']:.3f}, edge {z['edge']:.1e}, winding {z['winding']})")
            continue
        c, s, y = z["central"], z["stat"], z["sys"]
        print(f"    {c.real:6.3f} +- {s.real:.3f} +- {y.real:.3f}   {c.imag:6.3f} +- {s.imag:.3f} +- {y.imag:.3f}"
              f"   {abs(c):5.3f}  ({c.real * T_MEV:4.0f} + {c.imag * T_MEV:4.0f}i)"
              f"   {z['d_other']:6.3f}  {z['isolation']:5.2f}  {z['edge']:.1e}  {z['d_grid']:.1e}  {z['winding']:3d}"
              f"   {z['weight']:.3f}  {z['n_outliers']:3d}/{N_JK * z['n_variants']:<4d}  {'yes' if z['stable'] else 'no'}")
    stable = [z for z in cands if z["stable"]]
    if stable:
        r_min = min(abs(z["central"]) for z in stable)
        near = [z for z in stable if abs(z["central"]) <= IM_WINDOW * r_min]
        for z, what in [(min(near, key=lambda z: z["central"].imag), f"smallest Im mu among stable zeros with |mu| <= {IM_WINDOW} min|mu| (LY-zero candidate)"),
                        (min(stable, key=lambda z: abs(z["central"])), "smallest |mu| (radius of convergence of the Taylor series)")]:
            c, s, y = z["central"], z["stat"], z["sys"]
            print(f"  {what}:\n     mu_B/T = ({c.real:.3f} +- {np.hypot(s.real, y.real):.3f}) + ({c.imag:.3f} +- {np.hypot(s.imag, y.imag):.3f}) i,"
                  f"  |mu_B/T| = {abs(c):.3f},  mu_B = ({c.real * T_MEV:.0f} + {c.imag * T_MEV:.0f} i) MeV")
    else:
        print("  no stable zero in the search box")
    for z in zs:
        c = z.get("central", z["mu"])
        s = z.get("stat", np.nan + 1j * np.nan)
        y = z.get("sys", np.nan + 1j * np.nan)
        rows.append([tcode(n_max), c.real, s.real, y.real, c.imag, s.imag, y.imag, abs(c), z["d_other"], z["isolation"],
                     z["edge"], z.get("d_grid", np.nan), z.get("winding", 0), z.get("weight", np.nan),
                     z.get("n_outliers", 0), int(z["candidate"]), int(z["stable"])])

np.savetxt(OUT / f"lee_yang{TAG}.dat", np.array(rows),
           header=f"{'Nmax' if SCHEME == 'taylor' else 'sectors'}  Re  stat  sys  Im  stat  sys  |mu|  d_other  isolation  edge  d_grid  winding  weight  n_outliers  candidate  stable"
                  "   (mu = mu_B/T; errors only for candidates)",
           fmt=["%d"] + ["%.6e"] * 11 + ["%d", "%.3f", "%d", "%d", "%d"])

# ---- plot -------------------------------------------------------------------
re_m = np.arange(0, RE_MAX + MAP_STEP / 2, MAP_STEP)
im_m = np.arange(0, IM_MAX + MAP_STEP / 2, MAP_STEP)
mus_m = re_m[None, :] + 1j * im_m[:, None]
fig, axes = plt.subplots(1, len(N_MAX_LIST), figsize=(6 * len(N_MAX_LIST), 5.6), sharey=True)
for ax, n_max in zip(axes, N_MAX_LIST):
    lz = log_z_ratio(fits0_b, lr0_b[0], grid_b, mus_m, n_max).real
    pc = ax.pcolormesh(re_m, im_m, np.clip(lz, -10, 2), cmap="coolwarm", vmin=-10, vmax=2, shading="auto")
    for z in zeros[n_max]:
        if z["stable"]:
            c, s, y = z["central"], z["stat"], z["sys"]
            ax.errorbar(c.real, c.imag, xerr=np.hypot(s.real, y.real), yerr=np.hypot(s.imag, y.imag),
                        fmt="o", color="k", mfc="yellow", ms=7, capsize=3, lw=1.2, zorder=3)
        elif z["candidate"]:
            ax.plot(z["mu"].real, z["mu"].imag, "s", color="k", mfc="none", ms=7, zorder=3)
        else:
            ax.plot(z["mu"].real, z["mu"].imag, ".", color="k", ms=4, zorder=3)
    stable = [z for z in zeros[n_max] if z["stable"]]
    if stable:
        r_min = min(abs(z["central"]) for z in stable)
        cand = min([z for z in stable if abs(z["central"]) <= IM_WINDOW * r_min], key=lambda z: z["central"].imag)
        ax.plot(cand["central"].real, cand["central"].imag, "*", color="lime", mec="k", ms=16, zorder=4,
                label="LY-zero candidate (smallest Im)")
    ax.plot([], [], "o", color="k", mfc="yellow", ms=7, label="stable zero")
    ax.plot([], [], "s", color="k", mfc="none", ms=7, label="candidate that failed a check")
    ax.plot([], [], ".", color="k", ms=4, label="artifact zero")
    ax.set_title(f"{SCHEME} expansion, {tname(n_max)}")
    ax.set_xlabel(r"Re $\mu_B/T$")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    top = ax.secondary_xaxis("top", functions=(lambda x: x * T_MEV, lambda x: x / T_MEV))
    top.set_xlabel(r"Re $\mu_B$ [MeV]")
axes[0].set_ylabel(r"Im $\mu_B/T$")
right = axes[-1].secondary_yaxis("right", functions=(lambda y: y * T_MEV, lambda y: y / T_MEV))
right.set_ylabel(r"Im $\mu_B$ [MeV]")
fig.colorbar(pc, ax=axes, label=r"$\log|\mathcal{Z}(\mu_B)/\mathcal{Z}(0)|$", shrink=0.8)
fig.savefig(OUT / f"lee_yang{TAG}.png", dpi=150, bbox_inches="tight")
if "DOS_OUT" not in os.environ:
    plt.show()
