"""Plot the central values (with errors) from the .fit files.

Each .fit file has one row per x value:
    column 1   = x
    column 2   = mean
    column 3   = error
    columns 4+ = jackknife samples (not plotted)
Header lines that do not start with a number (e.g. '# chisq ...' or
'c Mean Err Jackknife') are skipped.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

here = Path(__file__).parent


def load(path):
    """Read a .fit file, skipping any header line that does not start with a number."""
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    with open(path) as f:
        rows = [line.split() for line in f if line.split() and is_number(line.split()[0])]
    return np.array(rows, dtype=float)

files = ["B2.fit", "B4.fit", "B6.fit", "B8.fit", "dlogrhodc.fit"]

fig, axes = plt.subplots(len(files), 1, figsize=(7, 2.6 * len(files)), sharex=True)

for ax, name in zip(axes, files):
    # keep only x, mean, error -- the jackknife columns are not plotted
    x, mean, err = load(here / name)[:, :3].T
    ax.errorbar(x, mean, yerr=err, fmt="o", ms=3, capsize=2)
    ax.set_ylabel(Path(name).stem)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("x")
fig.tight_layout()
fig.savefig(here / "plot.png", dpi=150)
plt.show()
