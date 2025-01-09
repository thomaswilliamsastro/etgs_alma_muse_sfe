import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from uncertainties import unumpy as unp

from vars import wisdom_dir, galaxies, plot_dir, sfr_dir, colours, dists, int_sfrs

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

sns.set_color_codes("deep")

os.chdir(wisdom_dir)

plot_name = os.path.join(plot_dir,
                         f"sfr_int_resolve_comparison",
                         )

fig, ax = plt.subplots(nrows=1, ncols=1,
                       figsize=(7, 5),
                       )
plt.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.7)

colour_dict = {}

for galaxy in galaxies:

    c = next(colours)

    if galaxy not in dists:
        continue

    dist = copy.deepcopy(dists[galaxy])

    colour_dict[galaxy] = copy.deepcopy(c)

    hdu_filename = os.path.join(sfr_dir,
                                f"{galaxy}_sfr_maps.fits",
                                )

    with fits.open(hdu_filename) as hdu:
        sfr_vals = copy.deepcopy(hdu["SFR"].data)
        sfr_errs = copy.deepcopy(hdu["eSFR"].data)

        sfr = unp.uarray(sfr_vals, sfr_errs)
        sn = copy.deepcopy(hdu["S2N"].data)
        bpt = copy.deepcopy(hdu["BPT"].data)
        w = WCS(hdu["SFR"])

        # Remove the logs
        sfr = 10 ** sfr

        # Turn to raw SFR measurement
        pixel_deg = wcs_utils.proj_plane_pixel_scales(w)[0]
        pixel_kpc = dist * 1e3 * np.radians(pixel_deg)
        sfr *= pixel_kpc ** 2

        # Take as an upper limit the sum of everything, and the lower limit the BPT cut

        all_mask = np.isfinite(sfr_vals)
        bpt_mask = bpt == 0 & all_mask

        # Skip if we've got nothing
        n_pix = len(all_mask[all_mask == True]) + len(bpt_mask[bpt_mask == True])
        if n_pix == 0:
            continue

        sfr_all = sfr[all_mask].sum()
        sfr_all = unp.log10(sfr_all)
        sfr_all = unp.nominal_values(sfr_all)

        # If there's nothing in the BPT mask at all, set this to a very low number
        sfr_bpt = sfr[bpt_mask].sum()
        if sfr_bpt == 0:
            sfr_bpt = unp.uarray(1e-99, 1)
        sfr_bpt = unp.log10(sfr_bpt)
        sfr_bpt = unp.nominal_values(sfr_bpt)

    if galaxy not in int_sfrs:
        continue

    int_sfr = copy.deepcopy(int_sfrs[galaxy])

    plt.plot([int_sfr.nominal_value, int_sfr.nominal_value],
             [sfr_all - int_sfr.nominal_value, sfr_bpt - int_sfr.nominal_value],
             color=c,
             lw=3,
             marker='none',
             label=galaxy.upper(),
             )

# xlim = plt.xlim()
# ylim = plt.ylim()
#
# ax_lim = np.nanmin([xlim[0], ylim[0]]), np.nanmax([xlim[1], ylim[1]])

plt.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 0.5),
    ncol=1,
    fancybox=False,
    edgecolor='k',
)

# Add in 1-1 line
# plt.plot(ax_lim, ax_lim,
#          c='k',
#          ls='--',
#          )
plt.axhline(0,
            c='k',
            ls='--',
            )

xlim = [-1.4, -0.25]
ylim = [-4.9, 0.4]

# Add in representative errorbar
plt.errorbar(xlim[0] + 0.3, ylim[1] - 0.2,
             xerr=0.2, yerr=0.1,
             c='k',
             ls='none',
             marker='none',
             lw=2,
             )
plt.xlim(xlim)
plt.ylim(ylim)

plt.minorticks_on()

ax.tick_params(axis="y", which="both", left=True, right=True)
ax.tick_params(axis="x", which="both", bottom=True, top=True)

plt.xlabel(r"$\log_{10}$(SFR$_\mathrm{FUV+WISE4}$ / $\mathrm{M}_\odot~\mathrm{yr}^{-1}$)")
plt.ylabel(r"$\log_{10}$(SFR$_\mathrm{H\alpha}$ / SFR$_\mathrm{FUV+WISE4}$)"
           )

plt.grid()

# plt.show()

plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')
plt.savefig(f"{plot_name}.png", bbox_inches='tight')

plt.close()

print("Complete!")
