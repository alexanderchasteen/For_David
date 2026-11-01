"""Scaling fits of the leading Lee-Yang zero mu_LY(T) and the critical endpoint they imply.

Input: tscan/summary.dat from ly_tscan.py (leading zero per T and truncation, mu_B in MeV).
Only temperatures where both truncations N_max = 8 and 6 have a candidate within MAX_DNMAX of
each other are used; the two truncations are fitted separately and their difference is the
truncation systematic. The temperatures are independent ensembles (uncorrelated errors).

Ansaetze. Near a critical point the QCD (T, mu) plane maps linearly onto the Ising (t, h)
plane, and the Yang-Lee edge sits at h = +- i h_0 t^(beta delta). Translated back this gives,
to leading order, a real part linear in T and an imaginary part with the edge exponent:

  M1  Ising edge         Re mu = mu_c + a (T - Tc),   Im mu = b (T - Tc)^(beta delta),   beta delta = 1.5645 (3D Ising)
  M2  Ising edge in mu^2 the same form for z = mu^2 (the natural variable of an even expansion); mu_c = sqrt(z_c)
  M3  M1 + finite volume Im mu = sqrt( b^2 (T - Tc)^(2 beta delta) + f^2 ):  in a finite box the edge
                         stops a distance f ~ L^(-beta delta / nu) above the real axis; f is a free parameter
                         here because there is a single volume
  M4  free exponent      M1 with beta delta fitted (consistency check: does the data prefer the Ising value?)

Fitting: for fixed (Tc, exponent) every model is linear in its remaining parameters, so those
are solved exactly and chi^2 is profiled over Tc (and the exponent for M4) on a grid.
Point errors: stat (+) sys from the fit variants, and, with TRUNC_SYS, (+) half the distance
between the N_max = 8 and 6 zeros of that temperature (the truncation uncertainty of the point;
without it chi^2/dof ~ 3 because the two truncations scatter more than their quoted errors).
Errors: Delta chi^2 = 1 of the profile for Tc; for the other parameters the linear-fit
covariance at the best Tc combined in quadrature with their variation across the Delta chi^2 <= 1
interval of Tc. Models are compared with the AIC (chi^2 + 2 n_params).
"""
import numpy as np
import matplotlib.pyplot as plt

from dos_common import here

BETA_DELTA = 1.5645
MAX_DNMAX = 100.0
TC_GRID = np.arange(-150.0, 128.0, 0.25)
EXP_GRID = np.arange(0.2, 3.01, 0.01)
TRUNC_SYS = True      # add half the N_max=8 vs 6 distance of each point in quadrature to its error

summary = np.loadtxt(here / "tscan" / "summary.dat", ndmin=2)
T8 = set(summary[(summary[:, 1] == 8) & (summary[:, 8] <= MAX_DNMAX), 0])
T6 = set(summary[(summary[:, 1] == 6) & (summary[:, 8] <= MAX_DNMAX), 0])
T_fit = np.array(sorted(T8 & T6))
print(f"temperatures used: {[int(t) for t in T_fit]} MeV  (both truncations have a candidate closer than {MAX_DNMAX:.0f} MeV)")


def wls(X, y, w):
    """Weighted least squares: coefficients, covariance, chi^2."""
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    cov = np.linalg.inv(Xw.T @ Xw)
    coef = cov @ Xw.T @ yw
    return coef, cov, np.sum((yw - Xw @ coef) ** 2)


def fit_at(model, Tc, bd, T, re, re_e, im, im_e):
    """Exact linear solve of a model at fixed Tc and exponent. Returns dict with chi2 and parameters."""
    dT = T - Tc
    if np.any(dT <= 0):
        return None
    out = {}
    # real part: mu_c + a dT  (all models)
    coef, cov, chi_re = wls(np.column_stack([np.ones_like(T), dT]), re, 1 / re_e ** 2)
    out.update(mu_c=coef[0], mu_c_err=np.sqrt(cov[0, 0]), a=coef[1], a_err=np.sqrt(cov[1, 1]))
    x = dT ** bd
    if model == "M3":
        # Im^2 = b^2 x^2 + f^2, linear in (b^2, f^2); sigma(Im^2) = 2 Im sigma
        coef, cov, chi_im = wls(np.column_stack([x ** 2, np.ones_like(x)]), im ** 2, 1 / (2 * im * im_e) ** 2)
        if coef[1] < 0:  # the floor cannot be negative: refit with f = 0
            coef1, cov1, chi_im = wls(x[:, None] ** 2, im ** 2, 1 / (2 * im * im_e) ** 2)
            coef, cov = np.array([coef1[0], 0.0]), np.diag([cov1[0, 0], 0.0])
        b2, f2 = coef
        out.update(b=np.sqrt(max(b2, 0)), f=np.sqrt(max(f2, 0)),
                   f_err=np.sqrt(cov[1, 1]) / (2 * np.sqrt(f2)) if f2 > 0 else np.sqrt(np.sqrt(cov[1, 1])))
        # chi^2 is recomputed in Im (not Im^2) so that all models are compared on the same footing
        chi_im = np.sum(((im - np.sqrt(b2 * x ** 2 + max(f2, 0))) / im_e) ** 2)
    else:
        coef, cov, chi_im = wls(x[:, None], im, 1 / im_e ** 2)
        out.update(b=coef[0], b_err=np.sqrt(cov[0, 0]), f=0.0)
    out["chi2"] = chi_re + chi_im
    return out


def profile(model, T, re, re_e, im, im_e):
    """Profile chi^2 over Tc (and the exponent for M4); returns best fit and the Tc profile."""
    exps = EXP_GRID if model == "M4" else [BETA_DELTA]
    best, prof = None, np.full(len(TC_GRID), np.inf)
    for bd in exps:
        for i, Tc in enumerate(TC_GRID):
            r = fit_at(model, Tc, bd, T, re, re_e, im, im_e)
            if r is None:
                continue
            prof[i] = min(prof[i], r["chi2"])
            if best is None or r["chi2"] < best["chi2"]:
                best = dict(r, Tc=Tc, bd=bd)
    # Delta chi^2 <= 1 interval of the Tc profile
    ok = TC_GRID[prof <= best["chi2"] + 1]
    best["Tc_lo"], best["Tc_hi"] = ok.min(), ok.max()
    best["Tc_err"] = 0.5 * (ok.max() - ok.min())
    # spread of mu_c across the Delta chi^2 <= 1 interval, added to its linear-fit error
    mus = [fit_at(model, Tc, best["bd"], T, re, re_e, im, im_e)["mu_c"] for Tc in ok]
    best["mu_c_err"] = np.hypot(best["mu_c_err"], 0.5 * (max(mus) - min(mus)))
    if model == "M4":
        bds = [bd for bd in exps for Tc in TC_GRID
               if (r := fit_at(model, Tc, bd, T, re, re_e, im, im_e)) is not None and r["chi2"] <= best["chi2"] + 1]
        best["bd_err"] = 0.5 * (max(bds) - min(bds)) if bds else np.nan
    return best, prof


MODELS = {"M1": "Ising edge", "M2": "Ising edge in mu^2", "M3": "Ising + finite-volume floor", "M4": "free exponent"}
N_PAR = {"M1": 4, "M2": 4, "M3": 5, "M4": 5}
results = {}
for n_max in (8, 6):
    s = summary[(summary[:, 1] == n_max) & np.isin(summary[:, 0], T_fit)]
    T, re, re_e, im, im_e = s[:, 0], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
    if TRUNC_SYS:
        re_e, im_e = np.hypot(re_e, s[:, 8] / 2), np.hypot(im_e, s[:, 8] / 2)
    print(f"\nN_max = {n_max}  ({len(T)} temperatures, {2 * len(T)} data points"
          f"{'; errors include half the N_max 8-6 distance' if TRUNC_SYS else ''})")
    for model in MODELS:
        if model == "M2":
            # z = mu^2: Re z = Re^2 - Im^2, Im z = 2 Re Im, errors by linear propagation
            z_re, z_im = re ** 2 - im ** 2, 2 * re * im
            z_re_e = 2 * np.hypot(re * re_e, im * im_e)
            z_im_e = 2 * np.hypot(im * re_e, re * im_e)
            best, prof = profile("M1", T, z_re, z_re_e, z_im, z_im_e)
            zc, zc_e = best["mu_c"], best["mu_c_err"]
            best["mu_c"], best["mu_c_err"] = (np.sqrt(zc), zc_e / (2 * np.sqrt(zc))) if zc > 0 else (np.nan, np.nan)
        else:
            best, prof = profile(model, T, re, re_e, im, im_e)
        best["dof"] = 2 * len(T) - N_PAR[model]
        best["aic"] = best["chi2"] + 2 * N_PAR[model]
        results[(n_max, model)] = (best, prof)
        extra = ""
        if model == "M3":
            extra = f",  floor f = {best['f']:.0f} +- {best['f_err']:.0f} MeV"
        if model == "M4":
            extra = f",  beta*delta = {best['bd']:.2f} +- {best['bd_err']:.2f} (Ising {BETA_DELTA})"
        print(f"  {model} {MODELS[model]:28s} Tc = {best['Tc']:6.1f} +- {best['Tc_err']:4.1f} MeV"
              f"  [{best['Tc_lo']:.0f}, {best['Tc_hi']:.0f}]   mu_c = {best['mu_c']:5.0f} +- {best['mu_c_err']:3.0f} MeV"
              f"   chi2/dof = {best['chi2']:5.1f}/{best['dof']}{extra}")
    aics = np.array([results[(n_max, m)][0]["aic"] for m in MODELS])
    w = np.exp(-(aics - aics.min()) / 2)
    w /= w.sum()
    print("  AIC weights: " + ", ".join(f"{m} {x:.2f}" for m, x in zip(MODELS, w)))

print("\ntruncation systematic (N_max 8 vs 6):")
for model in MODELS:
    b8, b6 = results[(8, model)][0], results[(6, model)][0]
    print(f"  {model}: dTc = {abs(b8['Tc'] - b6['Tc']):4.1f} MeV,  d mu_c = {abs(b8['mu_c'] - b6['mu_c']):4.0f} MeV")

# ---- save ----------------------------------------------------------------------
rows = [[n, i + 1, b["Tc"], b["Tc_err"], b["mu_c"], b["mu_c_err"], b["a"], b["b"], b["f"], b["bd"], b["chi2"], b["dof"]]
        for n in (8, 6) for i, m in enumerate(MODELS) for b in [results[(n, m)][0]]]
np.savetxt(here / "tscan" / "scaling_fit.dat", np.array(rows), fmt="%.6g",
           header="Nmax  model(1=M1 Ising,2=M2 mu^2,3=M3 floor,4=M4 free exp)  Tc  err  mu_c  err  a  b  f  beta*delta  chi2  dof"
                  "   [MeV units; T used: " + " ".join(f"{int(t)}" for t in T_fit) + "]")

# ---- plot ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
styles = {"M1": ("-", "C0"), "M2": ("--", "C2"), "M3": ("-.", "C1"), "M4": (":", "C4")}
for col, n_max in enumerate((8, 6)):
    s = summary[(summary[:, 1] == n_max) & np.isin(summary[:, 0], T_fit)]
    T, re, re_e, im, im_e = s[:, 0], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
    if TRUNC_SYS:
        re_e, im_e = np.hypot(re_e, s[:, 8] / 2), np.hypot(im_e, s[:, 8] / 2)
    ax, axp = axes[0, col], axes[1, col]
    ax.errorbar(T, re, yerr=re_e, fmt="o", color="C3", capsize=3, label=r"Re $\mu_{LY}$")
    ax.errorbar(T, im, yerr=im_e, fmt="s", color="k", capsize=3, label=r"Im $\mu_{LY}$")
    for model in MODELS:
        best, prof = results[(n_max, model)]
        ls, color = styles[model]
        Tt = np.linspace(best["Tc"], T.max() + 5, 300)
        dT = np.clip(Tt - best["Tc"], 0, None)
        if model == "M2":
            z = (best["mu_c"] ** 2 + best["a"] * dT) + 1j * best["b"] * dT ** best["bd"]
            mu = np.sqrt(z)
            re_t, im_t = mu.real, mu.imag
        else:
            re_t = best["mu_c"] + best["a"] * dT
            im_t = np.sqrt((best["b"] * dT ** best["bd"]) ** 2 + best["f"] ** 2)
        ax.plot(Tt, re_t, ls, color=color, lw=1.2)
        ax.plot(Tt, im_t, ls, color=color, lw=1.2,
                label=f"{model}: $T_c$ = {best['Tc']:.0f}({best['Tc_err']:.0f}), $\\mu_c$ = {best['mu_c']:.0f}({best['mu_c_err']:.0f}) MeV")
        ax.plot(best["Tc"], best["mu_c"], "*", color=color, mec="k", ms=13, zorder=5)
        axp.plot(TC_GRID, prof - best["chi2"], ls, color=color, label=f"{model} {MODELS[model]}")
    ax.set_title(rf"$N_{{\max}} = {n_max}$", fontsize=11)
    ax.set_xlabel("T [MeV]")
    ax.set_ylabel(r"$\mu_{LY}$ [MeV]")
    ax.set_ylim(0, 1500)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    axp.axhline(1, color="gray", lw=0.8, ls="--")
    axp.set_ylim(0, 10)
    axp.set_xlim(TC_GRID.min(), TC_GRID.max())
    axp.set_xlabel(r"$T_c$ [MeV]")
    axp.set_ylabel(r"$\Delta\chi^2(T_c)$ profile")
    axp.legend(fontsize=7)
    axp.grid(alpha=0.3)
fig.suptitle("Scaling fits of the leading Lee-Yang zero; stars = critical endpoint of each ansatz", fontsize=10)
fig.tight_layout()
fig.savefig(here / "tscan" / "scaling_fit.png", dpi=150)
plt.show()
