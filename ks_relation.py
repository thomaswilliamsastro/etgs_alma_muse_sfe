import copy
import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from astropy.table import Table
from matplotlib import gridspec
from reproject import reproject_interp
from scipy.stats import gaussian_kde
from uncertainties import unumpy as unp

from completeness_limits import get_co_completeness_limit, get_sfr_completeness_limit
from vars import wisdom_dir, alma_dir, sfr_dir, plot_dir, galaxies, alma_files, alpha_co, colours

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

sns.set_color_codes("deep")

# Fit from Bigiel+ (2008)
bigiel_n = 1.0
bigiel_n_err = 0.1
bigiel_a = -2.1
bigiel_a_err = 0.2

os.chdir(wisdom_dir)

# Points from Pessa+ (2021)
pessa_tab = Table.read("global_KS_datapoints_100pc_mod2.dat", format='ascii')
pessa_idx = np.where(pessa_tab["logSFR"] != -20)
pessa_tab = pessa_tab[pessa_idx]

pessa_tdep = pessa_tab["logMmol"] - pessa_tab["logSFR"]

plot_name = os.path.join(plot_dir,
                         'etg_ks')

fig = plt.figure(figsize=(12, 4))

spec = gridspec.GridSpec(ncols=2,
                         nrows=2,
                         wspace=0,
                         hspace=0,
                         height_ratios=[1, 0.25],
                         )
ax = fig.add_subplot(spec[0, 0])
resid = fig.add_subplot(spec[1, 0])

ax_ulim = fig.add_subplot(spec[0, 1])
resid_ulim = fig.add_subplot(spec[1, 1])

all_colours = itertools.cycle(sns.color_palette("deep"))

all_sfes = []
all_sfes_upper_lims = []

all_resids = []
all_resids_upper_lims = []

for galaxy in galaxies:

    print(galaxy)

    c = next(colours)

    # Skip if we don't have ALMA data
    if galaxy not in alma_files:
        continue

    sfr_filename = os.path.join(sfr_dir,
                                f"{galaxy}_sfr_maps.fits",
                                )

    alma_filename = alma_files[galaxy]

    co_filename = os.path.join(alma_dir,
                               galaxy,
                               f"{alma_filename}_mom0.fits"
                               )
    e_co_filename = os.path.join(alma_dir,
                                 galaxy,
                                 f"{alma_filename}_emom0.fits"
                                 )

    # Pull out SFR
    with fits.open(sfr_filename) as sfr_hdu:
        sfr = copy.deepcopy(sfr_hdu["SFR"].data)
        e_sfr = copy.deepcopy(sfr_hdu["eSFR"].data)
        sn = copy.deepcopy(sfr_hdu["S2N"].data)
        bpt = copy.deepcopy(sfr_hdu["BPT"].data)
        sfr_hdr = copy.deepcopy(sfr_hdu["SFR"].header)

        sn = np.array(sn, dtype=bool)

    # Pull out CO and error
    with fits.open(co_filename) as co_hdu:
        co_vals = copy.deepcopy(co_hdu[0].data)
        co_hdr = copy.deepcopy(co_hdu[0].header)

    with fits.open(e_co_filename) as e_co_hdu:
        e_co_vals = copy.deepcopy(e_co_hdu[0].data)

    # Get a representative error in the SFR value
    muse_filename = os.path.join("muse",
                                 "rebin",
                                 f"{galaxy.upper()}_MAPS.fits",
                                 )

    sfr_completeness_limit = get_sfr_completeness_limit(muse_filename)
    sfr_completeness_limit = np.log10(3 * sfr_completeness_limit)

    # rep_e_sfr_val = np.nanmedian(e_sfr[e_sfr != 0])  # TODO: This doesn't work

    co_cube_filename = os.path.join(alma_dir, galaxy, f"{alma_filename.replace('_strict', '')}.fits")

    co_completeness_limit = get_co_completeness_limit(co_cube_filename)
    co_completeness_limit = np.log10(3 * alpha_co * co_completeness_limit)

    # Replace 0 values in CO as 3-sigma upper limits
    co_zero_idx = np.where(co_vals <= 0)

    co_vals[co_zero_idx] = 3 * e_co_vals[co_zero_idx]
    upper_lim_co = np.zeros_like(co_vals)
    upper_lim_co[co_zero_idx] = 1

    # Convert to Sigma_gas
    co = unp.uarray(co_vals, e_co_vals)

    sigma_gas = co * alpha_co

    sigma_gas = unp.log10(sigma_gas)
    sigma_gas_hdu = fits.ImageHDU(data=unp.nominal_values(sigma_gas), header=co_hdr)
    e_sigma_gas_hdu = fits.ImageHDU(data=unp.std_devs(sigma_gas), header=co_hdr)
    upper_lim_co_hdu = fits.ImageHDU(data=upper_lim_co, header=co_hdr)

    # Reproject CO to the SFR map
    sigma_gas_reproj = reproject_interp(sigma_gas_hdu,
                                        sfr_hdr,
                                        return_footprint=False,
                                        )
    e_sigma_gas_reproj = reproject_interp(e_sigma_gas_hdu,
                                          sfr_hdr,
                                          return_footprint=False,
                                          )
    upper_lim_co_reproj = reproject_interp(upper_lim_co_hdu,
                                           sfr_hdr,
                                           order='nearest-neighbor',
                                           return_footprint=False,
                                           )

    upper_lim_co_reproj[np.isnan(upper_lim_co_reproj)] = 0
    upper_lim_co_reproj = np.array(upper_lim_co_reproj, dtype=bool)

    # Create some masks, first for pixels that pass the BPT and S/N cuts
    bpt_sn_sfr = np.logical_and(bpt == 0, sn)
    good_gas = ~upper_lim_co_reproj

    bpt_sn_sfr_good_gas = bpt_sn_sfr & good_gas

    # Next, we have pixels that pass the BPT but not the S/N cuts. We'll distinguish them
    # here and plot fainter
    bpt_sfr = np.logical_and(bpt == 0, ~sn)
    bpt_sfr_good_gas = bpt_sfr & good_gas

    # Next we're into upper limits, first those that pass the S/N cuts
    not_bpt_sn = np.logical_and(bpt != 0, sn)
    not_bpt_sn_good_gas = not_bpt_sn & good_gas

    # And then that fail S/N cuts
    not_bpt = np.logical_and(bpt != 0, ~sn)
    not_bpt_good_gas = not_bpt & good_gas

    # Plot the primo SFR values
    ax.errorbar(sigma_gas_reproj[bpt_sn_sfr_good_gas],
                sfr[bpt_sn_sfr_good_gas],
                xerr=e_sigma_gas_reproj[bpt_sn_sfr_good_gas],
                yerr=e_sfr[bpt_sn_sfr_good_gas],
                color=c,
                marker='none',
                ls='none',
                # label=galaxy.upper(),
                zorder=99,
                # alpha=0.5,
                alpha=1,
                )

    ax.errorbar(np.nan,
                np.nan,
                yerr=1,
                xerr=1,
                color=c,
                marker='none',
                ls='none',
                label=galaxy.upper(), )

    # And the representative errors
    ax.axhline(sfr_completeness_limit,
               color=c,
               ls='--',
               alpha=0.5,
               )
    ax.axvline(co_completeness_limit,
               color=c,
               ls='--',
               alpha=0.5,
               )

    bigiel_pred = np.log10(
                      10 ** bigiel_a * (10 ** (np.asarray(sigma_gas_reproj[bpt_sn_sfr_good_gas])) / 10) ** bigiel_n
                      / 1.36  # Account for Helium
                  )

    all_resids.extend(sfr[bpt_sn_sfr_good_gas] - bigiel_pred)

    resid.errorbar(sigma_gas_reproj[bpt_sn_sfr_good_gas],
                   sfr[bpt_sn_sfr_good_gas] - bigiel_pred,
                   xerr=e_sigma_gas_reproj[bpt_sn_sfr_good_gas],
                   yerr=e_sfr[bpt_sn_sfr_good_gas],
                   color=c,
                   marker='none',
                   ls='none',
                   # label=galaxy.upper(),
                   zorder=99,
                   # alpha=0.5,
                   alpha=1,
                   )

    # And the less primo SFR values
    ax.errorbar(sigma_gas_reproj[bpt_sfr_good_gas],
                sfr[bpt_sfr_good_gas],
                xerr=e_sigma_gas_reproj[bpt_sfr_good_gas],
                yerr=e_sfr[bpt_sfr_good_gas],
                color=c,
                marker='none',
                ls='none',
                # label=galaxy.upper(),
                zorder=98,
                alpha=0.1,
                )

    bigiel_pred = np.log10(
                      10 ** bigiel_a * (10 ** (np.asarray(sigma_gas_reproj[bpt_sfr_good_gas])) / 10) ** bigiel_n
                      / 1.36  # Account for Helium
                  )

    resid.errorbar(sigma_gas_reproj[bpt_sfr_good_gas],
                   sfr[bpt_sfr_good_gas] - bigiel_pred,
                   xerr=e_sigma_gas_reproj[bpt_sfr_good_gas],
                   yerr=e_sfr[bpt_sfr_good_gas],
                   color=c,
                   marker='none',
                   ls='none',
                   # label=galaxy.upper(),
                   zorder=98,
                   alpha=0.1,
                   )

    # Plot the pixels outside the SF BPT that pass S/N checks
    ax_ulim.errorbar(sigma_gas_reproj[not_bpt_sn_good_gas],
                     sfr[not_bpt_sn_good_gas],
                     xerr=e_sigma_gas_reproj[not_bpt_sn_good_gas],
                     yerr=e_sfr[not_bpt_sn_good_gas],
                     uplims=True,
                     color=c,
                     marker='none',
                     ls='none',
                     alpha=0.1,
                     )

    bigiel_pred = np.log10(
                      10 ** bigiel_a * (10 ** (np.asarray(sigma_gas_reproj[not_bpt_sn_good_gas])) / 10) ** bigiel_n
                      / 1.36  # Account for Helium
                  )

    all_resids_upper_lims.extend(sfr[not_bpt_sn_good_gas] - bigiel_pred)

    resid_ulim.errorbar(sigma_gas_reproj[not_bpt_sn_good_gas],
                        sfr[not_bpt_sn_good_gas] - bigiel_pred,
                        xerr=e_sigma_gas_reproj[not_bpt_sn_good_gas],
                        yerr=e_sfr[not_bpt_sn_good_gas],
                        uplims=True,
                        color=c,
                        marker='none',
                        ls='none',
                        # label=galaxy.upper(),
                        alpha=0.1,
                        )

    # And those that don't pass S/N cuts
    ax_ulim.errorbar(sigma_gas_reproj[not_bpt_good_gas],
                     sfr[not_bpt_good_gas],
                     xerr=e_sigma_gas_reproj[not_bpt_good_gas],
                     yerr=e_sfr[not_bpt_good_gas],
                     uplims=True,
                     color=c,
                     marker='none',
                     ls='none',
                     alpha=0.05,
                     rasterized=True,
                     )

    bigiel_pred = np.log10(
                      10 ** bigiel_a * (10 ** (np.asarray(sigma_gas_reproj[not_bpt_good_gas])) / 10) ** bigiel_n
                      / 1.36  # Account for Helium
                  )

    resid_ulim.errorbar(sigma_gas_reproj[not_bpt_good_gas],
                        sfr[not_bpt_good_gas] - bigiel_pred,
                        xerr=e_sigma_gas_reproj[not_bpt_good_gas],
                        yerr=e_sfr[not_bpt_good_gas],
                        uplims=True,
                        color=c,
                        marker='none',
                        ls='none',
                        # label=galaxy.upper(),
                        alpha=0.05,
                        rasterized=True,
                        )

    # Calculate the depletion times in years
    t_dep_sfe = 10 ** sigma_gas_reproj[bpt_sn_sfr_good_gas] * 1e6 / 10 ** sfr[bpt_sn_sfr_good_gas]
    t_dep_sfe = np.log10(t_dep_sfe)

    t_dep_sfe_upper_lim = 10 ** sigma_gas_reproj[not_bpt_sn_good_gas] * 1e6 / 10 ** sfr[not_bpt_sn_good_gas]
    t_dep_sfe_upper_lim = np.log10(t_dep_sfe_upper_lim)

    all_sfes.extend(t_dep_sfe)
    all_sfes_upper_lims.extend(t_dep_sfe_upper_lim)

print(f"Median residual for good pix: {np.nanmedian(all_resids)}")
print(f"Median residual for upper lims: {np.nanmedian(all_resids_upper_lims)}")

# Scatter on the points from Pessa+ as a background cloud
for a in [ax, ax_ulim]:
    a.scatter(pessa_tab["logMmol"] - 6,
              pessa_tab["logSFR"],
              c='grey',
              edgecolors='none',
              alpha=0.01,
              marker='.',
              zorder=-99,
              rasterized=True,
              )

xlim = ax.get_xlim()
ylim = ax.get_ylim()

xlim = [xlim[0] - 0.4, xlim[1]]
ylim = [ylim[0], 1.5]

# MC up some errorbars for the Bigiel fits to HERACLES
n_draws = 500
n_points = 500
bigiel_x_plot = np.linspace(*xlim, n_points)
bigiel_y_plot = np.zeros([n_draws, n_points])
for i in range(n_draws):
    mc_n = np.random.normal(loc=bigiel_n, scale=bigiel_n_err)
    mc_a = np.random.normal(loc=bigiel_a, scale=bigiel_a_err)
    # bigiel_y_plot[i, :] = mc_n * np.asarray(bigiel_x_plot) + mc_a
    bigiel_y_plot[i, :] = 10 ** mc_a * (10 ** (np.asarray(bigiel_x_plot)) / 10) ** mc_n
bigiel_y_plot = np.log10(bigiel_y_plot / 1.36)  # 1.36 here accounts for Helium

bigiel_upper, bigiel_med, bigiel_lower = np.nanpercentile(bigiel_y_plot, [16, 50, 84], axis=0)

# Depletion times
depletion_times = np.array([8, 9, 10])

for a in [ax, ax_ulim]:
    a.plot(bigiel_x_plot,
           bigiel_med,
           c='g',
           ls='--',
           label="Bigiel+ (2008)")
    a.fill_between(bigiel_x_plot,
                   bigiel_lower, bigiel_upper,
                   color='g',
                   alpha=0.25,
                   )

    # for depletion_percent in depletion_percents:
    #     a.plot(xlim, xlim + np.log10(depletion_percent) - 2,
    #            c='k',
    #            ls=':',
    #            )
    #     xpos = xlim[0] + 0.4
    #     a.text(xpos, xpos + np.log10(depletion_percent) - 2,
    #            f'{int(depletion_percent * 100)}%',
    #            ha='center', va='center',
    #            bbox=dict(facecolor='white', edgecolor='white', alpha=0.8),
    #            )
    for depletion_time in depletion_times:
        a.plot(xlim, np.asarray(xlim) + 6 - depletion_time,
               c='k',
               ls=':',
               )
        xpos = xlim[0] + 0.4
        a.text(xpos, xpos + 6 - depletion_time,
               f'$10^{{{depletion_time}}}$',
               ha='center', va='center',
               bbox=dict(facecolor='white', edgecolor='white', alpha=0.8),
               )

    a.set_xlim(xlim)
    a.set_ylim(ylim)

for r in [resid, resid_ulim]:
    r.plot(xlim, [0, 0],
           c='g',
           ls='--',
           )
    r.fill_between(bigiel_x_plot,
                   bigiel_lower - bigiel_med,
                   bigiel_upper - bigiel_med,
                   color='g',
                   alpha=0.25,
                   )

    r.set_xlim(xlim)
    r.set_ylim([-1.8, 1.8])

ax.scatter(-99, -99,
           c='grey',
           # edgecolor='none',
           marker='o',
           label="Pessa+ (2021)",
           )

# Factor of 0.8 here is because of the axis ratios
ax.legend(loc='center right',
          bbox_to_anchor=(-0.25, 0.5 * 0.8),
          ncol=1,
          fancybox=False,
          edgecolor='k',
          )

ax.minorticks_on()

ax.tick_params(axis="x", which="both", top=True)

ax_ulim.minorticks_on()

ax_ulim.tick_params(axis="x", which="both", top=True)

resid.minorticks_on()
resid_ulim.minorticks_on()

ax.text(0.05, 0.95,
        "SF BPT",
        ha='left',
        va='top',
        fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
        transform=ax.transAxes,
        )
ax_ulim.text(0.05, 0.95,
             "Non-SF BPT",
             ha='left',
             va='top',
             fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
             transform=ax_ulim.transAxes,
             )

ax.set_ylabel(r"$\log_{10}$($\Sigma_\mathrm{SFR}$ / $\mathrm{M}_\odot~\mathrm{yr}^{-1}~\mathrm{kpc}^{-2}$)")
ax_ulim.set_ylabel(r"$\log_{10}$($\Sigma_\mathrm{SFR}$ / $\mathrm{M}_\odot~\mathrm{yr}^{-1}~\mathrm{kpc}^{-2}$)",
                   rotation=-90,
                   labelpad=20,
                   )

ax_ulim.yaxis.set_label_position("right")
ax_ulim.yaxis.tick_right()

resid.set_xlabel(r"$\log_{10}$ ($\Sigma_\mathrm{mol}$ / $\mathrm{M}_\odot~\mathrm{pc}^{-2}$)")
resid.set_ylabel(r"$\Delta$")

resid_ulim.set_xlabel(r"$\log_{10}$ ($\Sigma_\mathrm{mol}$ / $\mathrm{M}_\odot~\mathrm{pc}^{-2}$)")
resid_ulim.set_ylabel(r"$\Delta$", rotation=-90, labelpad=20)
resid_ulim.yaxis.set_label_position("right")
resid_ulim.yaxis.tick_right()

ax.grid()
ax_ulim.grid()
resid.grid()
resid_ulim.grid()

plt.tight_layout()

# plt.show()

plt.savefig(f"{plot_name}.png", bbox_inches='tight', dpi=300)
plt.savefig(f"{plot_name}.pdf", bbox_inches='tight', dpi=300)
plt.close()

# Also make a KDE plot for the depletion times

plot_name = os.path.join(plot_dir,
                         't_deps')

good_idx = np.where(np.isfinite(all_sfes))
all_sfes = np.asarray(all_sfes)[good_idx]

good_idx = np.where(np.isfinite(all_sfes_upper_lims))
all_sfes_upper_lims = np.asarray(all_sfes_upper_lims)[good_idx]

med_all_sfes = np.nanmedian(all_sfes)
med_all_sfes_upper_lims = np.nanmedian(all_sfes_upper_lims)

bins = np.arange(7, 11.1, 0.01)

# Calculate the KDE and normalise to peak at 1
kde_sfe = gaussian_kde(all_sfes, bw_method="silverman")
kde_sfe_hist = kde_sfe.evaluate(bins)
kde_sfe_hist /= np.nanmax(kde_sfe_hist)

kde_pessa = gaussian_kde(pessa_tdep, bw_method="silverman")
kde_pessa_hist = kde_pessa.evaluate(bins)
kde_pessa_hist /= np.nanmax(kde_pessa_hist)

print(10 ** np.nanmedian(all_sfes))
print(10 ** np.nanmedian(pessa_tdep))

kde_sfe_upper_lims = gaussian_kde(all_sfes_upper_lims, bw_method="silverman")
kde_sfe_upper_lims_hist = kde_sfe_upper_lims.evaluate(bins)
kde_sfe_upper_lims_hist /= np.nanmax(kde_sfe_upper_lims_hist)

plt.figure(figsize=(7, 4))

ax = plt.subplot(1,1,1)

plt.plot(bins,
         kde_sfe_hist,
         c="k",
         )

plt.plot(bins,
         kde_sfe_upper_lims_hist,
         c="k",
         ls="--",
         alpha=0.5,
         )

plt.plot(bins,
         kde_pessa_hist,
         c="r",
         alpha=0.5,
         )

# Add in the medians of our distributions and the Bigiel one
plt.axvline(med_all_sfes,
            c="k",
            label="SF BPT",
            )
plt.axvline(med_all_sfes_upper_lims,
            c="k",
            ls="--",
            alpha=0.5,
            label="Non-SF BPT"
            )
plt.axvline(np.nanmedian(pessa_tdep),
            c="r",
            ls="-",
            alpha=0.5,
            label="Pessa+ (2021)"
            )
plt.axvline(np.log10(2e9),  # This already includes He per Bigiel
            c="g",
            ls="--",
            label="Bigiel+ (2008)"
            )

plt.xlim(bins[0], bins[-1])

plt.xlabel(r"$\log_{10}$($\tau_\mathrm{dep}$ / yr)")
plt.ylabel("Probability Density")

plt.minorticks_on()

ax.tick_params(axis="x", which="both", bottom=True, top=True)
ax.tick_params(axis="y", which="both", left=True, right=True)

plt.grid()

plt.legend(bbox_to_anchor=(1.05, 0.5),
           loc='center left',
           fancybox=False,
           edgecolor='k',
           )

plt.tight_layout()

# plt.show()

plt.savefig(f"{plot_name}.png", bbox_inches='tight')
plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')
plt.close()

print('Complete!')
