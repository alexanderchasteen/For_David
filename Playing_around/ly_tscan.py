"""Lee-Yang zero candidate as a function of temperature, and a Yang-Lee-edge estimate of
the critical endpoint.

    python ly_tscan.py [--scheme taylor|fugacity] [--rerun]

For every raw/<T>/ data set: check that the c range covers rho_0 (the sparse high-T
sets only sample a narrow window of c), run lee_yang.py on it (output in tscan/<T>/,
or tscan_<scheme>/<T>/) and collect the stable zeros. The candidate at each T is the
stable zero with the smallest Im mu near the smallest |mu| (notes, Sec. 2.3.1).
The two truncations compared are N_max = 8 and 6 (taylor) or 2 and 1 baryon sectors
(fugacity, both matched at N_max = 8, as in the notes' Fig. 3).

Critical endpoint: near a critical point the Yang-Lee edge approaches the real axis as
    Im mu_LY(T) = A (T - T_c)^(beta delta),        3D Ising: beta delta = 1.5645,
so T_c is found by a weighted least-squares fit of this form to Im mu_LY(T) (A is
linear and solved exactly; T_c by a 1-d minimisation of chi^2; error from
Delta chi^2 = 1). Re mu_LY(T) is extrapolated to T_c with a straight line. Both
truncations are fitted; their difference is the truncation systematic. Only temperatures
where both truncations have a candidate closer than MAX_DNMAX enter the fit.
The temperatures are independent ensembles, so their errors are uncorrelated.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

import dos_common as dc

EDGE_MAX = 1e-4       # max fraction of rho_0 allowed in the outer 5% of the c range
MIN_POINTS = 12
IM_WINDOW = 1.5       # candidate = smallest Im among stable zeros with |mu| <= IM_WINDOW * min |mu|
N_PARALLEL = 4
BETA_DELTA = 1.5645   # 3D Ising
RERUN = "--rerun" in sys.argv   # without --rerun, finished lee_yang*.dat files are reused
MAX_DNMAX = 100.0     # MeV; a temperature enters the CEP fit only if both truncations have a candidate this close to each other

SCHEME = sys.argv[sys.argv.index("--scheme") + 1] if "--scheme" in sys.argv else "taylor"
assert SCHEME in dc.SCHEMES, SCHEME
TRUNCS = (8, 6) if SCHEME == "taylor" else (2, 1)   # first column of lee_yang*.dat: N_max, or sectors kept
TNAME = "Nmax" if SCHEME == "taylor" else "sectors"
DAT = "lee_yang.dat" if SCHEME == "taylor" else f"lee_yang_{SCHEME}.dat"


def tlabel(tr, tex=True):
    if SCHEME == "taylor":
        return rf"$N_{{\max}} = {tr}$" if tex else f"N_max = {tr}"
    return f"{tr} sector{'s' if tr > 1 else ''}"


raw_root = dc.here / "raw"
out_root = dc.here / ("tscan" if SCHEME == "taylor" else f"tscan_{SCHEME}")
out_root.mkdir(exist_ok=True)
print(f"inner expansion scheme: {SCHEME}; truncations compared: {tlabel(TRUNCS[0], False)} and {tlabel(TRUNCS[1], False)}; "
      f"output in {out_root.name}/")

# ---- which temperatures have usable data ------------------------------------
usable = []
print("data sets:")
for d in sorted(raw_root.iterdir(), key=lambda d: int(d.name)):
    T = int(d.name)
    dc.configure(d)
    n = len(dc.c_raw)
    if n < MIN_POINTS:
        print(f"  T = {T:3d}: {n:3d} c values -> skipped (too few points)")
        continue
    variants, weights = dc.build_variants()
    edge, peak = dc.coverage(variants[int(np.argmax(weights))])
    ok = edge < EDGE_MAX and 0.1 < peak < 0.9
    print(f"  T = {T:3d}: {n:3d} c values in [{dc.c_raw[0]:g}, {dc.c_raw[-1]:g}], rho_0 at the edges {edge:.1e}, "
          f"peak at {peak:.2f} of the range -> {'run' if ok else 'skipped (range does not cover rho_0)'}")
    if ok:
        usable.append(T)


# ---- run lee_yang.py on each -------------------------------------------------
def run(T):
    out = out_root / str(T)
    out.mkdir(exist_ok=True)
    if not RERUN and (out / DAT).exists():
        return T, 0
    env = dict(os.environ, DOS_RAW=str(raw_root / str(T)), DOS_T=str(T), DOS_OUT=str(out), DOS_SCHEME=SCHEME,
               MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with open(out / "log.txt", "w") as log:
        r = subprocess.run([sys.executable, str(dc.here / "lee_yang.py")], env=env, stdout=log, stderr=subprocess.STDOUT)
    return T, r.returncode


print(f"\nrunning lee_yang.py for T = {usable} ({N_PARALLEL} at a time)", flush=True)
with ThreadPoolExecutor(N_PARALLEL) as pool:
    for T, rc in pool.map(run, usable):
        print(f"  T = {T}: {'done' if rc == 0 else f'FAILED (exit {rc}, see {out_root.name}/{T}/log.txt)'}", flush=True)

# ---- collect ----------------------------------------------------------------
# lee_yang*.dat columns: trunc Re stat sys Im stat sys |mu| d_other isolation edge d_grid winding weight n_outliers candidate stable
summary = []
print("\nLY-zero candidate (stable zero with the smallest Im mu near the smallest |mu|), mu_B in MeV:")
print(f"   T  {TNAME:>7s}  Re mu_B          Im mu_B          |mu_B/T|  Im mu_B/T  d_other[MeV]  stable zeros")
for T in usable:
    tab = np.loadtxt(out_root / str(T) / DAT, ndmin=2)
    for tr in TRUNCS:
        rows = tab[(tab[:, 0] == tr) & (tab[:, 16] == 1)]
        if len(rows) == 0:
            print(f"  {T:3d}  {tr:7d}  no stable zero")
            continue
        near = rows[rows[:, 7] <= IM_WINDOW * rows[:, 7].min()]   # same rule as lee_yang.py
        z = near[np.argmin(near[:, 4])]
        re_err, im_err = np.hypot(z[2], z[3]), np.hypot(z[5], z[6])
        print(f"  {T:3d}  {tr:7d}  {z[1] * T:6.0f} +- {re_err * T:4.0f}   {z[4] * T:6.0f} +- {im_err * T:4.0f}"
              f"   {z[7]:6.3f}   {z[4]:5.2f}    {z[8] * T:5.0f}         {len(rows)}")
        summary.append([T, tr, z[1] * T, re_err * T, z[4] * T, im_err * T, z[7], len(rows), z[8] * T])
summary = np.array(summary)
np.savetxt(out_root / "summary.dat", summary, fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%.4f", "%d", "%.2f"],
           header=f"T[MeV]  {TNAME}  Re_muB[MeV]  err  Im_muB[MeV]  err  |mu_B/T|  n_stable_zeros  dist_to_other_trunc[MeV]"
                  f"   (scheme {SCHEME}; candidate = stable zero with smallest Im among |mu| <= {IM_WINDOW} min|mu|)")


# ---- critical endpoint from the Yang-Lee edge --------------------------------
def edge_fit(T, im, err):
    """Fit im = A (T - Tc)^BETA_DELTA. Returns Tc, its error, A, chi2, dof."""
    def chi2_of(Tc):
        x = np.clip(T - Tc, 0, None) ** BETA_DELTA
        A = np.sum(x * im / err ** 2) / np.sum(x ** 2 / err ** 2)   # exact minimum in A
        return np.sum(((im - A * x) / err) ** 2), A

    Tcs = np.arange(-300, T.min() - 0.01, 0.05)
    chi = np.array([chi2_of(t)[0] for t in Tcs])
    i = int(np.argmin(chi))
    Tc, (chi_min, A) = Tcs[i], chi2_of(Tcs[i])
    inside = Tcs[chi <= chi_min + 1]
    return Tc, 0.5 * (inside.max() - inside.min()), A, chi_min, len(T) - 2


cep = {}
# temperatures where both truncations found a candidate and agree to MAX_DNMAX: only those
# are trusted for the extrapolation (elsewhere the highest term or the RW structure takes over)
Ta = set(summary[(summary[:, 1] == TRUNCS[0]) & (summary[:, 8] <= MAX_DNMAX), 0]) if len(summary) else set()
Tb = set(summary[(summary[:, 1] == TRUNCS[1]) & (summary[:, 8] <= MAX_DNMAX), 0]) if len(summary) else set()
T_fit = sorted(Ta & Tb)
print(f"\nYang-Lee edge extrapolation  Im mu_LY = A (T - Tc)^{BETA_DELTA} , using T = {[int(t) for t in T_fit]}"
      f" (both truncations have a candidate, closer than {MAX_DNMAX:.0f} MeV):")
for tr in TRUNCS:
    s = summary[(summary[:, 1] == tr) & np.isin(summary[:, 0], T_fit)] if len(summary) else summary
    if len(s) < 3:
        print(f"  {tlabel(tr, False)}: fewer than 3 usable temperatures, no fit")
        continue
    T, re_, re_e, im, im_e = s[:, 0], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
    Tc, dTc, A, chi2, dof = edge_fit(T, im, im_e)
    # straight line for Re mu_LY(T), extrapolated to Tc, with its error
    W = np.diag(1 / re_e ** 2)
    X = np.column_stack([np.ones_like(T), T])
    cov = np.linalg.inv(X.T @ W @ X)
    coef = cov @ X.T @ W @ re_
    muc = coef[0] + coef[1] * Tc
    dmuc = np.sqrt(np.array([1, Tc]) @ cov @ np.array([1, Tc]) + (coef[1] * dTc) ** 2)
    # robustness: drop the lowest or the highest temperature
    drops = [edge_fit(T[1:], im[1:], im_e[1:])[0], edge_fit(T[:-1], im[:-1], im_e[:-1])[0]]
    cep[tr] = (Tc, dTc, A, chi2, dof, muc, dmuc, coef)
    print(f"  {tlabel(tr, False)}: Tc = {Tc:.0f} +- {dTc:.0f} MeV  (chi2/dof = {chi2:.1f}/{dof}),  "
          f"Re mu_B(Tc) = {muc:.0f} +- {dmuc:.0f} MeV;  dropping lowest / highest T: Tc = {drops[0]:.0f} / {drops[1]:.0f} MeV")
if len(cep) == 2:
    a, b = TRUNCS
    print(f"  truncation systematic ({tlabel(a, False)} vs {tlabel(b, False)}): dTc = {abs(cep[a][0] - cep[b][0]):.0f} MeV, "
          f"d Re mu_B = {abs(cep[a][5] - cep[b][5]):.0f} MeV")

# ---- plots ------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
ax_re, ax_im, ax_r = axes
for tr, mfc, dx in [(TRUNCS[0], None, -0.6), (TRUNCS[1], "none", 0.6)]:
    s = summary[summary[:, 1] == tr] if len(summary) else summary
    if len(s) == 0:
        continue
    first = tr == TRUNCS[0]
    ax_re.errorbar(s[:, 0] + dx, s[:, 2], yerr=s[:, 3], fmt="o", color="C3", mfc=mfc or "C3", capsize=3, label=tlabel(tr))
    ax_im.errorbar(s[:, 0] + dx, s[:, 4], yerr=s[:, 5], fmt="s", color="C0", mfc=mfc or "C0", capsize=3, label=tlabel(tr))
    ax_r.plot(s[:, 0] + dx, s[:, 6], "D", color="k", mfc=mfc or "k", label=tlabel(tr))
    used = np.isin(s[:, 0], T_fit)
    ax_im.plot(s[~used, 0] + dx, s[~used, 4], "x", color="k", ms=9, mew=1.5, zorder=4,
               label="not used in the fit" if first else None)
    if tr in cep:
        Tc, dTc, A, chi2, dof, muc, dmuc, coef = cep[tr]
        Tt = np.linspace(Tc, s[:, 0].max() + 5, 200)
        ax_im.plot(Tt, A * (Tt - Tc) ** BETA_DELTA, "-" if first else "--", color="C0",
                   label=rf"$A(T-T_c)^{{\beta\delta}}$, $T_c = {Tc:.0f} \pm {dTc:.0f}$ MeV")
        ax_re.plot(Tt, coef[0] + coef[1] * Tt, "-" if first else "--", color="C3")
        ax_re.errorbar(Tc, muc, xerr=dTc, yerr=dmuc, fmt="*", ms=12, color="C3", mfc="yellow" if first else "none",
                       capsize=3, label=rf"CEP estimate: $\mu_B = {muc:.0f} \pm {dmuc:.0f}$ MeV")
ax_im.axhline(0, color="k", lw=0.6)
ax_r.axhline(np.pi, color="gray", ls=":", label=r"$\pi$ (Roberge-Weiss, $\mu_B/T = i\pi$)")
ax_re.set_ylabel(r"Re $\mu_{LY}$ [MeV]")
ax_im.set_ylabel(r"Im $\mu_{LY}$ [MeV]")
ax_r.set_ylabel(r"$|\mu_{LY}| / T$  (Taylor radius)")
ax_r.set_xlabel("T [MeV]")
ax_re.set_title(rf"Lee-Yang zero candidate vs T, {SCHEME} expansion  (stable zero with the smallest Im $\mu_B$; errors stat $\oplus$ sys)", fontsize=9)
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_root / "ly_tscan.png", dpi=150)
plt.show()
