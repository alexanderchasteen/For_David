"""Lee-Yang zero ratio R_21 of the two leading zeros, per temperature.

Following Wada, Kitazawa, Kanaya (arXiv:2501.18904, 2503.22246): order the zeros by
increasing imaginary part, z_1 the closest to the real axis, and form

    R_21 = Im z_2 / Im z_1 .

Expectations in the thermodynamic limit:
    first-order transition:  zeros equally spaced, Im z_n = a (2n-1)/V  ->  R_21 -> 3
    no transition (crossover / Lee-Yang edge at finite distance):     R_21 -> 1
    critical point:          R_21 -> universal number of the class, in between.
Here the zeros are those of the DoS-resummed Z(mu)/Z(0) in the Taylor scheme: the
first two candidates (zeros that passed the edge / truncation / isolation checks) with
|mu| within IM_WINDOW of the smallest, sorted by Im mu. Errors: stat (+) sys of each
zero, propagated as independent. Because the zeros lie along a curve rather than on an
axis, the ratio of Im(mu^2) (the even variable) is given as well.

Input: tscan/<T>/lee_yang.dat.  Output: tscan/ly_ratio.dat, tscan/ly_ratio.png.
"""
import numpy as np
import matplotlib.pyplot as plt

from dos_common import here

IM_WINDOW = 1.5
TEMPS = [130, 135, 140, 145, 150, 155, 160]
TRUNCS = (8, 6)

rows = []
print("   T  Nmax   z1 = Re + Im i (mu_B/T)        z2                       R_21 = Im z2/Im z1    Im(z2^2)/Im(z1^2)   |z2 - z1|   stable(z1,z2)")
for T in TEMPS:
    tab = np.loadtxt(here / "tscan" / str(T) / "lee_yang.dat", ndmin=2)
    for n_max in TRUNCS:
        c = tab[(tab[:, 0] == n_max) & (tab[:, 15] == 1)]          # candidates
        if len(c):
            c = c[c[:, 7] <= IM_WINDOW * c[:, 7].min()]             # near the smallest |mu|
        c = c[np.argsort(c[:, 4])]
        if len(c) < 2:
            print(f"  {T:3d}  {n_max}    only {len(c)} candidate zero(s)")
            continue
        z1, z2 = c[0], c[1]
        re1, im1, re2, im2 = z1[1], z1[4], z2[1], z2[4]
        e_re1, e_im1 = np.hypot(z1[2], z1[3]), np.hypot(z1[5], z1[6])
        e_re2, e_im2 = np.hypot(z2[2], z2[3]), np.hypot(z2[5], z2[6])
        R = im2 / im1
        dR = R * np.hypot(e_im2 / im2, e_im1 / im1)
        # ratio in mu^2: Im(z^2) = 2 Re Im
        q1, q2 = 2 * re1 * im1, 2 * re2 * im2
        dq1 = 2 * np.hypot(im1 * e_re1, re1 * e_im1)
        dq2 = 2 * np.hypot(im2 * e_re2, re2 * e_im2)
        Rq = q2 / q1
        dRq = Rq * np.hypot(dq2 / q2, dq1 / q1)
        print(f"  {T:3d}  {n_max}    {re1:5.3f} + {im1:5.3f} i          {re2:5.3f} + {im2:5.3f} i         {R:5.3f} +- {dR:5.3f}"
              f"          {Rq:5.3f} +- {dRq:5.3f}         {abs(complex(re2, im2) - complex(re1, im1)):5.3f}       {int(z1[16])},{int(z2[16])}")
        rows.append([T, n_max, re1, im1, re2, im2, R, dR, Rq, dRq, z1[16], z2[16]])
rows = np.array(rows)
np.savetxt(here / "tscan" / "ly_ratio.dat", rows, fmt=["%d", "%d"] + ["%.6f"] * 8 + ["%d", "%d"],
           header="T  Nmax  Re_z1  Im_z1  Re_z2  Im_z2  R21=Im_z2/Im_z1  err  Im(z2^2)/Im(z1^2)  err  stable_z1  stable_z2   (mu_B/T units)")

fig, ax = plt.subplots(figsize=(7, 4.5))
for n_max, mfc, dx in [(8, "C0", -0.5), (6, "none", 0.5)]:
    s = rows[rows[:, 1] == n_max]
    both = (s[:, 10] == 1) & (s[:, 11] == 1)
    ax.errorbar(s[both, 0] + dx, s[both, 6], yerr=s[both, 7], fmt="o", color="C0", mfc=mfc, capsize=3,
                label=rf"$R_{{21}} = \mathrm{{Im}}\,z_2 / \mathrm{{Im}}\,z_1$, $N_{{\max}} = {n_max}$")
    ax.errorbar(s[~both, 0] + dx, s[~both, 6], yerr=s[~both, 7], fmt="o", color="C0", mfc=mfc, mec="gray", capsize=3, alpha=0.5)
    ax.errorbar(s[both, 0] + dx, s[both, 8], yerr=s[both, 9], fmt="s", color="C1", mfc=mfc if mfc == "none" else "C1", capsize=3,
                label=rf"$\mathrm{{Im}}\,z_2^2 / \mathrm{{Im}}\,z_1^2$, $N_{{\max}} = {n_max}$")
ax.axhline(3, color="C3", ls="--", lw=1, label="first order: 3")
ax.axhline(1, color="k", ls=":", lw=1, label="crossover: 1")
ax.set_xlabel("T [MeV]")
ax.set_ylabel("ratio of the two leading zeros")
ax.set_ylim(0.8, 3.3)
ax.set_title("Lee-Yang zero ratio (Wada-Kitazawa-Kanaya); faded = one of the two zeros failed a stability check", fontsize=9)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(here / "tscan" / "ly_ratio.png", dpi=150)
plt.show()
