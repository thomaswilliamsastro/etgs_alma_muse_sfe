import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from scipy.stats import gaussian_kde

from vars import wisdom_dir, galaxies, plot_dir, sfr_dir

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

sns.set_color_codes("deep")

os.chdir(wisdom_dir)

plot_name = os.path.join(plot_dir,
                         f"sig_halpha_distributions",
                         )

all_vel_disps = []
sfr_vel_disps = []
vel_errs = []

for galaxy in galaxies:
    hdu_filename = os.path.join(sfr_dir,
                                f"{galaxy}_sfr_maps.fits",
                                )
    maps_filename = os.path.join("muse",
                                 "rebin",
                                 f"{galaxy.upper()}_MAPS.fits",
                                 )

    with fits.open(hdu_filename) as hdu:
        bpt = copy.deepcopy(hdu["BPT"].data)

    with fits.open(maps_filename) as maps_hdu:
        sig_ha = copy.deepcopy(maps_hdu["HA6562_SIGMA"].data)
        corr = copy.deepcopy(maps_hdu["HA6562_SIGMA_CORR"].data)

        sig_ha_corr = np.sqrt(sig_ha ** 2 - corr ** 2)
        # sig_ha_corr[sig_ha_corr > 100] = np.nan

        vel_errs.extend(maps_hdu["HA6562_SIGMA_ERR"].data.flatten())

    # all_vel_disps.extend(sig_ha_corr[bpt != 0 & np.isfinite(bpt)])
    all_vel_disps.extend(sig_ha_corr[bpt != 0 & np.isfinite(sig_ha_corr)])
    sfr_vel_disps.extend(sig_ha_corr[bpt == 0 & np.isfinite(bpt)])

all_vel_disps = np.array(all_vel_disps)
# all_vel_disps = np.log10(all_vel_disps)
all_vel_disps = all_vel_disps[np.isfinite(all_vel_disps)]

sfr_vel_disps = np.array(sfr_vel_disps)
# sfr_vel_disps = np.log10(sfr_vel_disps)
sfr_vel_disps = sfr_vel_disps[np.isfinite(sfr_vel_disps)]

print(np.nanpercentile(sfr_vel_disps, [16, 50, 84]))
print(np.nanpercentile(all_vel_disps, [16, 50, 84]))
print(np.nanmedian(vel_errs))

bins = np.arange(0, 500, 1)

# KDE these bad boys up
all_vel_disps_kde = gaussian_kde(all_vel_disps, bw_method="silverman")
all_vel_disps_hist = all_vel_disps_kde.evaluate(bins)
all_vel_disps_hist /= np.nanmax(all_vel_disps_hist)

sfr_vel_disps_kde = gaussian_kde(sfr_vel_disps, bw_method="silverman")
sfr_vel_disps_hist = sfr_vel_disps_kde.evaluate(bins)
sfr_vel_disps_hist /= np.nanmax(sfr_vel_disps_hist)

plt.figure(figsize=(5, 4))
ax = plt.subplot(1, 1, 1)

plt.plot(bins,
         all_vel_disps_hist,
         c="k",
         label="Non-SF BPT",
         )

plt.plot(bins,
         sfr_vel_disps_hist,
         c="b",
         label="SF BPT",
         )

plt.xlim(bins[0], bins[-1])
plt.ylim(0, 1.1)

plt.xlabel(r"$\sigma_{\mathrm{H}\alpha}$ (km s$^{-1}$)")
plt.ylabel(r"Probability Density")

plt.legend(loc='upper right',
           fancybox=False,
           edgecolor='k',
           )

plt.minorticks_on()

ax.tick_params(axis="x", which="both", bottom=True, top=True)
ax.tick_params(axis="y", which="both", left=True, right=True)

plt.grid()

plt.tight_layout()

plt.savefig(f"{plot_name}.pdf", bbox_inches="tight")
plt.savefig(f"{plot_name}.png", bbox_inches="tight")
plt.close()

# plt.show()

print("Complete!")
