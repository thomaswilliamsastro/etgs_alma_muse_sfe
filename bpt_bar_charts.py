import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits

from vars import wisdom_dir, galaxies, plot_dir, sfr_dir, dists

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

sns.set_color_codes("deep")

os.chdir(wisdom_dir)

plot_name = os.path.join(plot_dir,
                         f"bpt_barchart",
                         )

# Set up arrays for the figure
categories = [
    "SF",
    "Comp.",
    "Sy",
    "LIER",
]
category_colours = [
    "b",
    "y",
    "g",
    "r",
]
fractions = {}

for galaxy in galaxies:

    if galaxy not in dists:
        continue

    muse_hdu_name = os.path.join(sfr_dir,
                                 f"{galaxy}_sfr_maps.fits",
                                 )

    dist = copy.deepcopy(dists[galaxy])

    with fits.open(muse_hdu_name) as muse_hdu:

        bpt = copy.deepcopy(muse_hdu["BPT"].data)

        # We want to take a mask of all valid values
        pix_mask = np.isfinite(bpt)

        n_pix = np.nansum(pix_mask)

        fracs = []
        for i in range(len(categories)):
            fracs.append(len(bpt[bpt == i]) / n_pix)

        fractions[galaxy.upper()] = fracs

labels = list(fractions.keys())
data = np.array(list(fractions.values()))
data_cum = data.cumsum(axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.invert_yaxis()
ax.set_xlim(0, 1)

for i, (colname, colour) in enumerate(zip(categories, category_colours)):
    widths = data[:, i]
    starts = data_cum[:, i] - widths
    rects = ax.barh(labels, widths, left=starts, height=0.5,
                    label=colname, color=colour)

ax.legend(bbox_to_anchor=(1.05, 0.5),
          loc='center left',
          fancybox=False,
          edgecolor='k',
          )

plt.minorticks_on()
ax.tick_params(axis="y", which="both", left=False)
ax.tick_params(axis="x", which="both", bottom=True, top=True)
plt.grid(axis="x")
ax.set_xlabel(f"$f$")

plt.tight_layout()

plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')
plt.savefig(f"{plot_name}.png", bbox_inches='tight')

# plt.show()
plt.close()

