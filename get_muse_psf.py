import copy
import os

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.table import Table

from musepsf.image import Image
from musepsf.musepsf import MUSEImage
from vars import wisdom_dir, galaxies, sdss_ref_dir, muse_r_band_dir, sdss_psf_dir, psf_match_dir, plot_dir

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

sns.set_color_codes("deep")

os.chdir(wisdom_dir)

if not os.path.exists(psf_match_dir):
    os.makedirs(psf_match_dir)

fit_alpha = False

plot_name = os.path.join(plot_dir,
                         "muse_psf")
if fit_alpha:
    plot_name += "_fit_alpha"

psf_tab = os.path.join(psf_match_dir,
                       "psf_tab.fits",
                       )

best_fit_psfs = []
best_fit_alphas = []
fit_galaxies = []
ao_mode = []

muse_seeing = {
    "ngc0524": (0.65 + 0.61 + 0.88 + 0.82) / 4,
    "ngc1317": (0.67 + 0.62 + 0.65 + 0.75) / 4,
    "ngc3489": np.nan,
    "ngc3599": (0.56 + 0.57 + 0.65 + 0.53) / 4,
    "ngc3607": (0.76 + 0.69 + 0.86 + 0.84) / 4,
    "ngc3626": np.nan,
    "ngc4435": (0.57 + 0.59 + 0.72 + 0.86) / 4,
    "ngc4457": (0.72 + 0.7 + 0.94 + 0.73) / 4,
    "ngc4596": (0.86 + 0.66 + 0.76 + 0.99) / 4,
    "ngc4694": (0.83 + 0.47 + 0.63 + 0.65) / 4,
    "ngc4697": (0.81 + 0.83 + 1.15 + 0.72) / 4,
    "ngc7743": (0.66 + 0.7 + 0.59 + 0.71) / 4,
}

muse_airmass = {
    "ngc0524": (1.276 + 1.249 + 1.233 + 1.219) / 4,
    "ngc1317": (1.402 + 1.326 + 1.28 + 1.229) / 4,
    "ngc3489": np.nan,
    "ngc3599": (1.375 + 1.362 + 1.358 + 1.359) / 4,
    "ngc3607": (1.372 + 1.36 + 1.356 + 1.359) / 4,
    "ngc3626": np.nan,
    "ngc4435": (1.267 + 1.279 + 1.293 + 1.319) / 4,
    "ngc4457": (1.154 + 1.141 + 1.136 + 1.133) / 4,
    "ngc4596": (1.277 + 1.311 + 1.343 + 1.393) / 4,
    "ngc4694": (1.253 + 1.275 + 1.298 + 1.333) / 4,
    "ngc4697": (1.479 + 1.391 + 1.334 + 1.271) / 4,
    "ngc7743": (1.297 + 1.336 + 1.373 + 1.43) / 4,
}

for galaxy in galaxies:
    ref_name = os.path.join(f"{galaxy}_r.fits")
    psf_name = os.path.join(sdss_psf_dir,
                            f"{galaxy}_r.psf.fits")
    muse_name = os.path.join(f"{galaxy}_muse_r.fits")

    # If we don't have a PSF, move on
    if not os.path.exists(psf_name):
        continue

    print(f"Fitting {galaxy}")

    sdss = Image(ref_name,
                 input_dir=sdss_ref_dir,
                 output_dir=psf_match_dir)
    muse = MUSEImage(muse_name,
                     input_dir=muse_r_band_dir,
                     output_dir=psf_match_dir,
                     )

    # Keep track of whether it's AO mode or not
    if muse.main_header['HIERARCH ESO TPL ID'] == "MUSE_wfm-ao_obs_genericoffsetLGS":
        ao_mode.append(1)
    else:
        ao_mode.append(0)

    equivalency = u.spectral_density(6231 * u.AA)

    sdss.convert_units(muse.units, equivalency=equivalency)

    # Load in the PSF
    sdss.open_psf(psf_name)

    new_sdss = copy.deepcopy(sdss)
    muse.measure_psf(reference=new_sdss,
                     fit_alpha=fit_alpha,
                     plot=True,
                     save=True,
                     show=False,
                     edge=50,
                     )

    best_fit_psfs.append(muse.best_fit[0])
    if fit_alpha:
        best_fit_alphas.append(muse.best_fit[1])
    fit_galaxies.append(galaxy)

ao_mode = np.array(ao_mode)

ao_idx = ao_mode == 1

best_fit_psfs = np.array(best_fit_psfs)
log_seeing = np.array([muse_seeing[galaxy] for galaxy in fit_galaxies])
log_airmass = np.array([muse_airmass[galaxy] for galaxy in fit_galaxies])

# Normalise alpha between 0 and 1
log_airmass_alpha = (log_airmass - np.nanmin(log_airmass)) / (np.nanmax(log_airmass) - np.nanmin(log_airmass))
log_airmass_alpha[np.isnan(log_airmass_alpha)] = 0

plt.figure(figsize=(5, 4))
# plt.scatter(best_fit_psfs[np.where(ao_idx)[0]],
#             log_seeing[np.where(ao_idx)[0]],
#             color='k',
#             alpha=log_airmass_alpha[np.where(ao_idx)[0]],
#             label='AO',
#             )
plt.scatter(best_fit_psfs[np.where(~ao_idx)[0]],
            log_seeing[np.where(~ao_idx)[0]],
            color='r',
            alpha=log_airmass_alpha[np.where(~ao_idx)[0]],
            label='No AO',
            )

xlim = plt.xlim()
ylim = plt.ylim()

lim = np.nanmin([xlim[0], ylim[0]]), np.nanmax([xlim[1], ylim[1]])

plt.plot(lim, lim,
         c='k',
         ls='--',
         )

plt.xlim(lim)
plt.ylim(lim)

plt.minorticks_on()
plt.grid()

plt.legend(loc='upper left',
           fancybox=False,
           )

plt.xlabel("Fit PSF (arcsec)")
plt.ylabel("DIMM (arcsec)")

plt.tight_layout()

plt.show()

if fit_alpha:
    figsize = (8, 4)
else:
    figsize = (5, 4)

plt.figure(figsize=figsize)

if fit_alpha:
    plt.subplot(1, 2, 1)
else:
    plt.subplot(1, 1, 1)

plt.hist(best_fit_psfs,
         color='k',
         lw=2,
         histtype='step')

plt.minorticks_on()
plt.grid()

plt.xlabel(r"FWHM (arcsec)")
plt.ylabel(r"$N$")

if fit_alpha:
    ax = plt.subplot(1, 2, 2)

    plt.hist(best_fit_alphas,
             color='k',
             lw=2,
             histtype='step')

    plt.minorticks_on()
    plt.grid()

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    plt.xlabel(r"$\alpha$ (arcsec[?])")
    plt.ylabel(r"$N$")

plt.tight_layout()

plt.subplots_adjust(hspace=0, wspace=0.05)

plt.show()

plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')
plt.savefig(f"{plot_name}.png", bbox_inches='tight')

plt.close()

tab = Table(data=[fit_galaxies, best_fit_psfs],
            names=['galaxy', 'psf_arcsec'])
tab.write(psf_tab)

print("Complete!")
