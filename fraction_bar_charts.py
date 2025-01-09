import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from reproject import reproject_interp, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs

from vars import wisdom_dir, galaxies, plot_dir, sfr_dir, dists, alma_dir, alma_files

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

sns.set_color_codes("deep")

os.chdir(wisdom_dir)

use_all_halpha = True

plot_name = os.path.join(plot_dir,
                         f"overlap_barchart",
                         )

if use_all_halpha:
    plot_name += "_all_halpha"

# Set up arrays for the figure
categories = [r"$f_\mathrm{mol}$", r"$f_\mathrm{overlap}$"]

if use_all_halpha:
    categories.append(r"$f_\mathrm{H\alpha}$")
else:
    categories.append(r"$f_\mathrm{SFR}$")

category_colours = ["b", "gold", "r"]
fractions = {}

for galaxy in galaxies:

    if galaxy not in dists:
        continue

    muse_hdu_name = os.path.join(sfr_dir,
                                 f"{galaxy}_sfr_maps.fits",
                                 )

    alma_hdu_name = f"{alma_files[galaxy]}_mom0.fits"
    alma_hdu_name = alma_hdu_name.replace("_150pc", "")
    alma_hdu_name = os.path.join(alma_dir, galaxy, alma_hdu_name)

    dist = copy.deepcopy(dists[galaxy])

    with fits.open(muse_hdu_name) as muse_hdu, fits.open(alma_hdu_name) as alma_hdu:

        sfr_vals = copy.deepcopy(muse_hdu["SFR"].data)
        bpt = copy.deepcopy(muse_hdu["BPT"].data)

        # Get out optimal WCS and reproject
        wcs, shape = find_optimal_celestial_wcs([muse_hdu["SFR"], alma_hdu[0]])

        alma_reproj = reproject_exact(alma_hdu,
                                      wcs,
                                      shape_out=shape,
                                      return_footprint=False,
                                      )

        muse_sfr_reproj = reproject_exact(muse_hdu["SFR"],
                                          wcs,
                                          shape_out=shape,
                                          return_footprint=False,
                                          )

        muse_bpt_reproj = reproject_interp(muse_hdu["BPT"],
                                           wcs,
                                           shape_out=shape,
                                           return_footprint=False,
                                           order='nearest-neighbor',
                                           )

        # If we're using Halpha, take the whole BPT. Else take
        # just the SF bit of the BPT
        if use_all_halpha:
            muse_sfr_reproj[~np.isfinite(muse_bpt_reproj)] = 0
        else:
            muse_sfr_reproj[muse_bpt_reproj != 0] = 0

        # We want to take a mask of all valid values, and then ones that either have CO or SFR in them
        pix_mask = np.isfinite(alma_reproj) & np.isfinite(muse_sfr_reproj)
        pix_mask = pix_mask & np.logical_or(alma_reproj != 0, muse_sfr_reproj != 0)

        n_pix = np.nansum(pix_mask)

        f_co = np.logical_and(alma_reproj != 0, muse_sfr_reproj == 0) & pix_mask
        f_co = np.sum(f_co) / n_pix

        f_sfr = np.logical_and(alma_reproj == 0, muse_sfr_reproj != 0) & pix_mask
        f_sfr = np.sum(f_sfr) / n_pix

        f_overlap = 1 - f_co - f_sfr

        fractions[galaxy.upper()] = [f_co, f_overlap, f_sfr]

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
