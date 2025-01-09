from spectral_cube import SpectralCube
import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from scipy.stats import gaussian_kde
import warnings
import astropy.units as u
from reproject import reproject_interp

from external.sun_cube_tools import calc_channel_corr, censoring_function

from vars import wisdom_dir, galaxies, plot_dir, sfr_dir, alma_files, alma_dir, alpha_co

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

sns.set_color_codes("deep")

os.chdir(wisdom_dir)

plot_name = os.path.join(plot_dir,
                         f"alpha_vir_distributions",
                         )

sfr_alpha_vir = []
all_alpha_vir = []

for galaxy in galaxies:
    print(galaxy)

    alma_filename = alma_files[galaxy]

    # print('Calculating channel-to-channel correlation')

    # Read in cube, calculate the channel correlation
    mask_file_name = os.path.join(alma_dir, galaxy,
                                  alma_filename.replace("_strict", "_strictmask.fits"))
    with fits.open(mask_file_name) as mask_hdu:
        cube_mask = mask_hdu[0].data.astype(bool)

    cube_file_name = os.path.join(alma_dir, galaxy,
                                  alma_filename.replace("_strict", ".fits"))
    cube = SpectralCube.read(cube_file_name)

    # Get the velocity resolution (km/s)
    vel_res_int = np.abs(np.nanmedian(np.diff(cube.spectral_axis)))
    vel_res_int = vel_res_int.to(u.km / u.s).value

    channel_corr_mask = ~cube_mask
    channel_corr_mask[np.isnan(cube.hdu.data)] = False

    r, _ = calc_channel_corr(cube, mask=channel_corr_mask)

    # Read in velocity dispersion, and reproject the BPT mask to this guy
    vel_disp_filename = os.path.join(alma_dir,
                                     galaxy,
                                     f"{alma_filename}_ew.fits",
                                     )
    bpt_filename = os.path.join(sfr_dir,
                                f"{galaxy}_sfr_maps.fits",
                                )

    with fits.open(vel_disp_filename) as vel_disp_hdu, fits.open(bpt_filename) as bpt_hdu:
        vel_disp = vel_disp_hdu[0].data

        bpt = reproject_interp(bpt_hdu["BPT"],
                               vel_disp_hdu[0].header,
                               order="nearest-neighbor",
                               return_footprint=False,
                               )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        k = 0.47 * r - 0.23 * r ** 2 - 0.16 * r ** 3 + 0.43 * r ** 4
        sigma_resp = vel_res_int / np.sqrt(2 * np.pi) * (1 + 1.8 * k + 10.4 * k ** 2)

        vel_disp = np.sqrt(vel_disp ** 2 - sigma_resp ** 2)

    # Read in mom0 and convert to surf dens
    mom0_filename = os.path.join(alma_dir,
                                 galaxy,
                                 f"{alma_filename}_mom0.fits",
                                 )
    with fits.open(mom0_filename) as mom0_hdu:
        surf_dens = mom0_hdu[0].data * alpha_co

    r_beam = 150 / 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha_vir = 5.77 * vel_disp ** 2 * surf_dens ** -1 * (r_beam / 40) ** -1

    sfr_alpha_vir.extend(alpha_vir[bpt == 0 & np.isfinite(alpha_vir)])
    all_alpha_vir.extend(alpha_vir[bpt != 0 & np.isfinite(alpha_vir)])

all_alpha_vir = np.array(all_alpha_vir)
all_alpha_vir = all_alpha_vir[np.isfinite(all_alpha_vir)]

sfr_alpha_vir = np.array(sfr_alpha_vir)
sfr_alpha_vir = sfr_alpha_vir[np.isfinite(sfr_alpha_vir)]

print(np.nanpercentile(sfr_alpha_vir, [16, 50, 84]))
print(np.nanpercentile(all_alpha_vir, [16, 50, 84]))

bins = np.arange(0, 30, 0.03)

# KDE these bad boys up
all_alpha_vir_kde = gaussian_kde(all_alpha_vir, bw_method="silverman")
all_alpha_vir_hist = all_alpha_vir_kde.evaluate(bins)
all_alpha_vir_hist /= np.nanmax(all_alpha_vir_hist)

sfr_alpha_vir_kde = gaussian_kde(sfr_alpha_vir, bw_method="silverman")
sfr_alpha_vir_hist = sfr_alpha_vir_kde.evaluate(bins)
sfr_alpha_vir_hist /= np.nanmax(sfr_alpha_vir_hist)

plt.figure(figsize=(5, 4))
ax = plt.subplot(1, 1, 1)

plt.plot(bins,
         all_alpha_vir_hist,
         c="k",
         label="Non-SF BPT",
         )

plt.plot(bins,
         sfr_alpha_vir_hist,
         c="b",
         label="SF BPT",
         )

# Add on the Sun+2020 lines
plt.axvline(2.7,
            c='r',
            label="Sun+(2020)")
plt.axvline(10 ** (np.log10(2.7) - 0.7),
            c='r',
            ls="--",
            )
plt.axvline(10 ** (np.log10(2.7) + 0.7),
            c='r',
            ls="--",
            )

plt.xlim(bins[0], bins[-1])
plt.ylim(0, 1.1)

plt.xlabel(r"$\alpha_\mathrm{vir}$")
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

# plt.show()

plt.savefig(f"{plot_name}.pdf", bbox_inches="tight")
plt.savefig(f"{plot_name}.png", bbox_inches="tight")
plt.close()

print("Complete!")
