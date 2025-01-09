import copy
import os
import socket

import astropy.units as u
import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyneb as pn
import seaborn as sns
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from mpl_toolkits.axes_grid1 import make_axes_locatable
from uncertainties import unumpy as unp

from vars import wisdom_dir, galaxies, plot_dir, sfr_dir, ew_dir, colours

if 'mac' in socket.gethostname():
    os.environ['PATH'] += ':/Library/TeX/texbin'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'text.latex.preamble': r'\usepackage{txfonts}'})

os.chdir(wisdom_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(sfr_dir):
    os.makedirs(sfr_dir)

sns.set_color_codes("deep")

lines = [
    "HA6562",
    "HB4861",
    "NII6583",
    "OIII5006",
    "SII6716",
    "SII6730",
]

bpt_colour = {"sf": "b",
              "comp": "y",
              "sy": "g",
              "lier": "r",
              }
bpt_label = {"sf": "SF",
             "comp": "Comp.",
             "sy": "Sy",
             "lier": "LIER",
             }

whan_colour = {
    "sf": "b",
    "strong_agn": "y",
    "weak_agn": "g",
    "rg": "orange",
    "passive": "r",
}
whan_label = {
    "sf": "SF",
    "strong_agn": "sAGN",
    "weak_agn": "wAGN",
    "rg": "R",
    "passive": "P",
}

# S/N limit
sn_limit = 3  # UP FROM 2
c_hb_limit = 3

# Conversion from ergs/s from Kennicutt+Evans (2012)
sfr_fact = 41.27

all_oiii_hb = {}
all_nii_ha = {}
all_sii_ha = {}
all_nii_ha_whan = {}
all_ew = {}

for galaxy in galaxies:

    print(galaxy)

    hdu_name = os.path.join("muse",
                            "rebin",
                            f"{galaxy.upper()}_MAPS.fits",
                            )
    ew_name = os.path.join(ew_dir,
                           f"{galaxy.upper()}_ew.fits")

    with fits.open(hdu_name) as hdu, fits.open(ew_name) as ew_hdu:
        w = WCS(hdu[1].header)
        # for h in hdu[1:]:
        #     print(h.header['EXTNAME'])

        out_hdu = fits.HDUList()
        out_fits_filename = os.path.join(sfr_dir,
                                         f"{galaxy}_sfr_maps.fits",
                                         )

        # Pull out where there's data for a contour later
        obs_extent = np.zeros_like(hdu["ID"].data)
        obs_extent[np.isfinite(hdu["ID"].data)] = 1
        jj, ii = np.meshgrid(np.arange(obs_extent.shape[1]), np.arange(obs_extent.shape[0]))

        # Pull out Halpha for plotting later
        ha = copy.deepcopy(hdu["HA6562_FLUX"].data)

        line_dict = {}
        for line in lines:
            line_dict[line] = {}

        # Get out reddening corrections
        rc = pn.RedCorr(R_V=3.1, law='CCM89 oD94')
        for line in line_dict:
            lam = float(line[-4:])
            line_dict[line]["k"] = rc._CCM89_oD94(lam)

        # Pull out line maps we'll need
        for line in line_dict:
            flux = copy.deepcopy(hdu[f"{line}_FLUX"].data)
            flux_err = copy.deepcopy(hdu[f"{line}_FLUX_ERR"].data)

            # Mask out any zero fluxes
            idx = np.where(flux <= 0)
            flux[idx] = np.nan
            flux_err[idx] = np.nan

            line_dict[line]["map"] = unp.uarray(flux, flux_err)

        # Also pull the EW and error values out into an array
        ew_vals = unp.uarray(ew_hdu["EW"].data,
                             ew_hdu["EW_ERR"].data,
                             )

        # Create low S/N mask, for any of the strong lines
        sn_mask = np.zeros_like(ha, dtype=bool)
        for line in line_dict:
            sn = unp.nominal_values(line_dict[line]["map"]) / unp.std_devs(line_dict[line]["map"])
            sn_mask[sn < sn_limit] = True

        # Get rid of any unphysical observed ratios
        obs_ratio = line_dict["HA6562"]["map"] / line_dict["HB4861"]["map"]

        idx = np.where(obs_ratio < 2.86)
        for line in line_dict:
            line_dict[line]["map"][idx] = np.nan

        # Create map of Balmer decrement
        c = (np.log10(2.86) - unp.log10(obs_ratio)) / (0.4 * (line_dict["HA6562"]["k"] - line_dict["HB4861"]["k"]))

        # Set to 0 anything <0, as this is unphysical. Roll this into the S/N mask
        sn_mask[c < 0] = True
        c[c < 0] = 0

        # Set anything with absurdly high Balmer decrements to NaN
        c[c > c_hb_limit] = np.nan

        # Correct lines
        for line in line_dict:
            line_dict[line]["map_corr"] = line_dict[line]["map"] * 10 ** (0.4 * c * line_dict[line]["k"])

        # Roll unphysical values into the S/N mask
        for line in line_dict:
            map_corr = copy.deepcopy(line_dict[line]["map_corr"])
            line_unphysical_idx = map_corr == 0
            sn_mask[line_unphysical_idx] = True

        nii_ha = unp.log10(line_dict["NII6583"]["map_corr"] / line_dict["HA6562"]["map_corr"])
        sii_ha = unp.log10((line_dict["SII6716"]["map_corr"] + line_dict["SII6730"]["map_corr"]) /
                           line_dict["HA6562"]["map_corr"])
        oiii_hb = unp.log10(line_dict["OIII5006"]["map_corr"] / line_dict["HB4861"]["map_corr"])

        all_nii_ha[galaxy] = copy.deepcopy(unp.nominal_values(nii_ha))
        all_sii_ha[galaxy] = copy.deepcopy(unp.nominal_values(sii_ha))
        all_oiii_hb[galaxy] = copy.deepcopy(unp.nominal_values(oiii_hb))

        # Implement the various Kewley cuts
        sf_mask = (
                (oiii_hb < 1.3 + 0.61 / (nii_ha - 0.05)) &
                (nii_ha < 0.05) &
                (oiii_hb < 1.3 + 0.72 / (sii_ha - 0.32)) &
                (sii_ha < 0.32)
        )
        # sf_mask = sf_mask & ~sn_mask

        comp_mask = (
                (oiii_hb > 1.3 + 0.61 / (nii_ha - 0.05)) &
                (nii_ha < 0.05) &
                (oiii_hb < 1.19 + 0.61 / (nii_ha - 0.47)) &
                (nii_ha < 0.47)
        )
        # comp_mask = comp_mask & ~sn_mask

        sy_mask = (
                (oiii_hb > 1.19 + 0.61 / (nii_ha - 0.47)) &
                (nii_ha < 0.47) &
                (oiii_hb > 1.30 + 0.72 / (sii_ha - 0.32)) &
                (sii_ha < 0.32) &
                (oiii_hb > 0.76 + 1.89 * sii_ha)
        )
        # sy_mask = sy_mask & ~sn_mask

        lier_mask = (
                (oiii_hb > 1.19 + 0.61 / (nii_ha - 0.47)) &
                (nii_ha < 0.47) &
                (oiii_hb > 1.30 + 0.72 / (sii_ha - 0.32)) &
                (sii_ha < 0.32) &
                (oiii_hb < 0.76 + 1.89 * sii_ha)
        )
        # lier_mask = lier_mask & ~sn_mask

        # Calculate the various demarkation lines
        # nii_ha_lims = [-1, 0.1]
        nii_ha_lims = [-1, 0.3]
        sii_ha_lims = [-1.3, 0.4]
        oiii_hb_lims = [-2, 1.3]

        nii_ha_plot = np.linspace(*nii_ha_lims, num=100)
        sii_ha_plot = np.linspace(*sii_ha_lims, num=100)
        oiii_hb_plot = np.linspace(*oiii_hb_lims, num=100)

        oiii_hb_kauffmann_sf = 0.61 / (nii_ha_plot - 0.05) + 1.3
        oiii_hb_kauffmann_sf[nii_ha_plot > 0.05] = np.nan

        oiii_hb_kewley_sf = 0.61 / (nii_ha_plot - 0.47) + 1.19
        oiii_hb_kewley_sf[nii_ha_plot > 0.47] = np.nan

        sii_ha_kewley_sf = 0.72 / (sii_ha_plot - 0.32) + 1.3
        sii_ha_kewley_sf[sii_ha_plot > 0.32] = np.nan

        sii_ha_kewley_agn = 1.89 * sii_ha_plot + 0.76
        sii_ha_kewley_agn_idx = np.where(sii_ha_kewley_agn < sii_ha_kewley_sf)
        sii_ha_kewley_agn[sii_ha_kewley_agn_idx] = np.nan

        bpt_masks = {
            "sf": sf_mask,
            "comp": comp_mask,
            "sy": sy_mask,
            "lier": lier_mask,
        }

        # Get masks based on the WHAN diagram
        whan_sf_mask = (nii_ha < -0.4) & (ew_vals > 3)
        whan_sagn_mask = (nii_ha > -0.4) & (ew_vals > 6)
        whan_wagn_mask = (nii_ha > -0.4) & (ew_vals > 3) & (ew_vals < 6)
        whan_rg_mask = (ew_vals < 3) & (ew_vals > 0.5)
        whan_pg_mask = ew_vals < 0.5

        whan_masks = {
            "sf": whan_sf_mask,
            "strong_agn": whan_sagn_mask,
            "weak_agn": whan_wagn_mask,
            "rg": whan_rg_mask,
            "passive": whan_pg_mask,
        }

        nii_ha_vals = unp.nominal_values(nii_ha)
        sii_ha_vals = unp.nominal_values(sii_ha)
        oiii_hb_vals = unp.nominal_values(oiii_hb)

        # Also pull out typical errors
        nii_ha_err = unp.std_devs(nii_ha)
        nii_ha_err = nii_ha_err[nii_ha_err != 0]
        nii_ha_err = np.nanmean(nii_ha_err)

        sii_ha_err = unp.std_devs(sii_ha)
        sii_ha_err = sii_ha_err[sii_ha_err != 0]
        sii_ha_err = np.nanmean(sii_ha_err)

        oiii_hb_err = unp.std_devs(oiii_hb)
        oiii_hb_err = oiii_hb_err[oiii_hb_err != 0]
        oiii_hb_err = np.nanmean(oiii_hb_err)

        ew_vals = unp.nominal_values(ew_vals)

        # Convert from ergs/s/cm^2/spaxel to ergs/s/kpc^2
        arcsec_pixel = wcs_utils.proj_plane_pixel_scales(w)[0] * 3600
        conv_fact = 10 ** -20 * 4 * np.pi * (u.kpc.to(u.cm)) ** 2 / (arcsec_pixel * u.arcsec.to(u.rad)) ** 2

        # Create full SFR map, removing any pesky infs
        ha_corr = copy.deepcopy(line_dict["HA6562"]["map_corr"])
        ha_corr *= conv_fact
        # ha_corr[~np.isfinite(ha_corr)] = np.nan
        sfr = unp.log10(ha_corr) - sfr_fact

        sfr_vals = unp.nominal_values(sfr)
        sfr_err = unp.std_devs(sfr)

        # Add an SFR extension to the output fits file
        sfr_hdr = copy.deepcopy(hdu[1].header)
        sfr_hdr['EXTNAME'] = "SFR"
        sfr_hdr['BUNIT'] = 'log(MSUN/YR/KPC^2)'
        sfr_hdu = fits.PrimaryHDU(data=sfr_vals, header=sfr_hdr)

        out_hdu.append(sfr_hdu)

        # Add an SFR error extension to the output fits file
        sfr_err_hdr = copy.deepcopy(hdu[1].header)
        sfr_err_hdr['EXTNAME'] = "eSFR"
        sfr_err_hdr['BUNIT'] = 'log(MSUN/YR/KPC^2)'
        sfr_err_hdu = fits.ImageHDU(data=sfr_err, header=sfr_err_hdr)

        out_hdu.append(sfr_err_hdu)

        # Create a SN map
        sn_map = np.zeros_like(ha)
        sn_map[~sn_mask] = 1

        # Add in SN extension to the output fits
        sn_hdr = copy.deepcopy(hdu[1].header)
        sn_hdr['EXTNAME'] = "S2N"

        sn_hdu = fits.ImageHDU(data=sn_map, header=sn_hdr)

        out_hdu.append(sn_hdu)

        # Create a BPT map
        bpt_map = np.ones_like(ha) * np.nan

        for i, key in enumerate(bpt_masks.keys()):
            bpt_map[bpt_masks[key]] = i

        # Add in BPT extension to the output fits
        bpt_hdr = copy.deepcopy(hdu[1].header)
        bpt_hdr['EXTNAME'] = "BPT"
        bpt_hdr["COMMENT"] = "Labels:"
        bpt_hdr["COMMENT"] = "0: SF"
        bpt_hdr["COMMENT"] = "1: COMP"
        bpt_hdr["COMMENT"] = "2: SY"
        bpt_hdr["COMMENT"] = "3: LIER"

        bpt_hdu = fits.ImageHDU(data=bpt_map, header=bpt_hdr)

        out_hdu.append(bpt_hdu)

        # Create a WHAN map
        whan_map = np.ones_like(ha) * np.nan

        for i, key in enumerate(whan_masks.keys()):
            whan_map[whan_masks[key]] = i

        # Add in BPT extension to the output fits
        whan_hdr = copy.deepcopy(hdu[1].header)
        whan_hdr['EXTNAME'] = "WHAN"
        whan_hdr["COMMENT"] = "Labels:"
        whan_hdr["COMMENT"] = "0: SF"
        whan_hdr["COMMENT"] = "1: sAGN"
        whan_hdr["COMMENT"] = "2: wAGN"
        whan_hdr["COMMENT"] = "3: RG"
        whan_hdr["COMMENT"] = "4: PG"

        whan_hdu = fits.ImageHDU(data=whan_map, header=whan_hdr)

        out_hdu.append(whan_hdu)

        out_hdu.writeto(out_fits_filename, overwrite=True)

        # Mask out anything not defined as star-forming
        sfr_vals[~sf_mask] = np.nan
        sfr_vals[sn_mask] = np.nan

        vmin_sfr, vmax_sfr = np.nanpercentile(sfr_vals, [1, 99])
        vmin_ha, vmax_ha = np.nanpercentile(ha[ha != 0], [2, 98])

        plot_name = os.path.join(plot_dir,
                                 f"{galaxy}_bpt",
                                 )

        fig, axs = plt.subplots(nrows=1, ncols=2,
                                figsize=(8, 5),
                                squeeze=False,
                                )

        axs[0, 0].scatter(nii_ha_vals,
                          oiii_hb_vals,
                          color='grey',
                          s=1,
                          alpha=0.25,
                          )

        # Add in representative errorbar
        print(nii_ha_err)
        # axs[0, 0].errorbar(nii_ha_lims[0] + 0.1, oiii_hb_lims[0] + 0.1,
        #                    xerr=nii_ha_err, yerr=oiii_hb_err,
        #                    c='k',
        #                    ls='none',
        #                    marker='none',
        #                    lw=2,
        #                    )

        for key in bpt_masks.keys():
            axs[0, 0].scatter(nii_ha_vals[bpt_masks[key] & ~sn_mask],
                              oiii_hb_vals[bpt_masks[key] & ~sn_mask],
                              color=bpt_colour[key],
                              s=1,
                              alpha=0.5,
                              )

        axs[0, 0].plot(nii_ha_plot, oiii_hb_kauffmann_sf,
                       c='k',
                       lw=1,
                       label="Kauffmann+ (2003)",
                       )

        axs[0, 0].plot(nii_ha_plot, oiii_hb_kewley_sf,
                       c='k',
                       lw=1,
                       ls="--",
                       label="Kewley+ (2006; SF)",
                       )
        axs[0, 0].plot(-99, -99,
                       c='k',
                       lw=1,
                       ls="-.",
                       label="Kewley+ (2006; Sy/LIER)",
                       )

        axs[0, 0].set_xlim(nii_ha_lims)
        axs[0, 0].set_ylim(oiii_hb_lims)

        axs[0, 0].axes.minorticks_on()

        axs[0, 0].xaxis.set_ticks_position('both')

        axs[0, 0].axes.grid()

        axs[0, 0].legend(loc='lower center',
                         bbox_to_anchor=(0.5, 1.02),
                         edgecolor="k",
                         fancybox=False,
                         )

        axs[0, 0].set_xlabel(r'$\log_{10}$([N{\sc ii}]$\lambdaup 6583$/H$\alphaup$)')
        axs[0, 0].set_ylabel(r'$\log_{10}$([O{\sc iii}]$\lambdaup 5006$/H$\betaup$)')

        axs[0, 1].scatter(sii_ha_vals,
                          oiii_hb_vals,
                          color='grey',
                          s=1,
                          alpha=0.25,
                          # label="All",
                          )
        axs[0, 1].scatter(-99, -99,
                          color='grey',
                          label="All",
                          )

        for key in bpt_masks.keys():
            axs[0, 1].scatter(sii_ha_vals[bpt_masks[key] & ~sn_mask],
                              oiii_hb_vals[bpt_masks[key] & ~sn_mask],
                              color=bpt_colour[key],
                              s=1,
                              alpha=0.5,
                              )

            axs[0, 1].scatter(-99, -99,
                              color=bpt_colour[key],
                              label=bpt_label[key],
                              )

        axs[0, 1].plot(sii_ha_plot, sii_ha_kewley_sf,
                       c='k',
                       lw=1,
                       ls="--",
                       # label="Kewley et al. (2008)",
                       )
        axs[0, 1].plot(sii_ha_plot, sii_ha_kewley_agn,
                       c='k',
                       lw=1,
                       ls="-.",
                       # label="Kewley et al. (2008)",
                       )

        axs[0, 1].set_xlim(sii_ha_lims)
        axs[0, 1].set_ylim(oiii_hb_lims)

        axs[0, 1].axes.minorticks_on()

        axs[0, 1].xaxis.set_ticks_position('both')

        axs[0, 1].yaxis.set_label_position("right")
        axs[0, 1].yaxis.tick_right()

        axs[0, 1].axes.grid()

        axs[0, 1].legend(loc='lower center',
                         bbox_to_anchor=(0.5, 1.02),
                         ncol=2,
                         edgecolor="k",
                         fancybox=False,
                         )

        axs[0, 1].set_xlabel(r'$\log_{10}$([S{\sc ii}]$\lambdaup\lambdaup 6716,30$/H$\alphaup$)')
        axs[0, 1].set_ylabel(r'$\log_{10}$([O{\sc iii}]$\lambdaup 5006$/H$\betaup$)', rotation=-90, labelpad=20)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)

        plt.show()

        plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')
        plt.savefig(f"{plot_name}.png", bbox_inches='tight')

        plt.close()

        # Also get the WHAN plot, with map on left and parameter space on right

        plot_name = os.path.join(plot_dir,
                                 f"{galaxy}_whan",
                                 )

        fig = plt.figure(figsize=(8, 5))

        whan_xlims = [-1, 0.6]
        whan_ylims = [0.1, 115]

        # Put the WHAN on a map
        ax = plt.subplot(1, 2, 1, projection=w)

        # ax.contour(jj, ii,
        #            obs_extent,
        #            colors='k',
        #            linewidths=0.5,
        #            levels=1,
        #            )

        cmap = copy.deepcopy(cmocean.cm.gray)
        cmap.set_bad(color='black')

        ax.imshow(ha,
                  vmin=vmin_ha,
                  vmax=vmax_ha,
                  origin='lower',
                  cmap=cmap,
                  alpha=0.5,
                  interpolation='nearest',
                  )

        whan_cmap = matplotlib.colors.ListedColormap([whan_colour[key] for key in whan_colour.keys()])
        whan_cmap.set_bad("white", alpha=0)

        whan_map[sn_mask] = np.nan

        im = ax.imshow(whan_map,
                       vmin=-0.5,
                       vmax=len(whan_colour.keys()) - 0.5,
                       origin='lower',
                       cmap=whan_cmap,
                       interpolation='nearest',
                       )

        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec. (J2000)")

        ax.coords[0].display_minor_ticks(True)
        ax.coords[1].display_minor_ticks(True)

        plt.grid()

        plt.text(x=0.05, y=0.95,
                 s=galaxy.upper(),
                 ha='left', va='top',
                 fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='black'),
                 transform=ax.transAxes,
                 )

        # Get the aspect ratio for the purposes of getting the right plot shape
        # later
        ax1_ratio = ax.get_data_ratio()

        ax = plt.subplot(1, 2, 2)

        all_nii_ha_whan[galaxy] = copy.deepcopy(nii_ha_vals)
        all_ew[galaxy] = copy.deepcopy(ew_vals)

        ax.scatter(nii_ha_vals,
                   ew_vals,
                   color='grey',
                   s=1,
                   alpha=0.25,
                   )
        ax.scatter(-99, -99,
                   color='grey',
                   label="All",
                   )

        for key in whan_masks.keys():
            ax.scatter(nii_ha_vals[whan_masks[key] & ~sn_mask],
                       ew_vals[whan_masks[key] & ~sn_mask],
                       color=whan_colour[key],
                       s=1,
                       alpha=0.5,
                       )

            ax.scatter(-99, -99,
                       color=whan_colour[key],
                       label=whan_label[key],
                       )

        # Add on the various demarkation lines
        ax.axhline(0.5,
                   lw=1,
                   c='k',
                   ls='--',
                   )

        ax.axhline(3,
                   lw=1,
                   c='k',
                   ls='--',
                   )

        ax.hlines(6,
                  xmin=-0.4,
                  xmax=100,
                  lw=1,
                  colors='k',
                  ls='--',
                  )

        ax.vlines(-0.4,
                  ymin=3,
                  ymax=1e8,
                  lw=1,
                  colors='k',
                  ls='--',
                  )

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles,
                   labels,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 0.95),
                   ncol=3,
                   edgecolor="k",
                   fancybox=False,
                   )

        ax.set_xlim(whan_xlims)
        ax.set_ylim(whan_ylims)

        ax.set_yscale("log")

        # ax.axes.grid()

        ax.set_xlabel(r'$\log_{10}$([N{\sc ii}]$\lambdaup 6583$/H$\alphaup$)')
        ax.set_ylabel(r"W$_\mathrm{H\alphaup}$ ($\AA$)", rotation=-90, labelpad=20)

        ax.axes.minorticks_on()

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.tick_params(axis="y", which="both", left=True, right=True)
        ax.tick_params(axis="x", which="both", bottom=True, top=True)

        # Add in some labels
        ax.text(x=-0.95, y=10 ** (np.log10(0.1) + 0.1),
                s="P",
                ha='left', va='bottom',
                fontweight='bold',
                fontsize=14,
                # bbox=dict(facecolor='white', edgecolor='black'),
                )
        ax.text(x=-0.95, y=10 ** (np.log10(0.5) + 0.1),
                s="R",
                ha='left', va='bottom',
                fontweight='bold',
                fontsize=14,
                # bbox=dict(facecolor='white', edgecolor='black'),
                )
        ax.text(x=-0.95, y=10 ** (np.log10(3) + 0.1),
                s="SF",
                ha='left', va='bottom',
                fontweight='bold',
                fontsize=14,
                # bbox=dict(facecolor='white', edgecolor='black'),
                )
        ax.text(x=0.52, y=10 ** (np.log10(3) + 0.05),
                s="wAGN",
                ha='right', va='bottom',
                fontweight='bold',
                fontsize=14,
                # bbox=dict(facecolor='white', edgecolor='black'),
                )
        ax.text(x=0.52, y=10 ** (np.log10(100) - 0.1),
                s="sAGN",
                ha='right', va='top',
                fontweight='bold',
                fontsize=14,
                # bbox=dict(facecolor='white', edgecolor='black'),
                )

        ax.set_aspect(ax1_ratio / ax.get_data_ratio())

        # plt.show()

        plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')
        plt.savefig(f"{plot_name}.png", bbox_inches='tight')

        plt.close()

        plot_name = os.path.join(plot_dir,
                                 f"{galaxy}_sfr_map",
                                 )

        plt.figure(figsize=(12, 6),
                   )

        # Add in SFR map
        ax = plt.subplot(1, 2, 1, projection=w)

        # ax.contour(jj, ii,
        #            obs_extent,
        #            colors='k',
        #            linewidths=0.5,
        #            levels=1,
        #            )

        cmap = copy.deepcopy(cmocean.cm.gray)
        cmap.set_bad(color='black')

        ax.imshow(ha,
                  vmin=vmin_ha,
                  vmax=vmax_ha,
                  origin='lower',
                  # cmap=cmocean.cm.gray,
                  cmap=cmap,
                  alpha=0.5,
                  interpolation='nearest',
                  )

        sfr_cmap = cmocean.cm.thermal
        sfr_cmap.set_bad("white", alpha=0)

        im = ax.imshow(sfr_vals,
                       vmin=vmin_sfr,
                       vmax=vmax_sfr,
                       origin='lower',
                       cmap=sfr_cmap,
                       interpolation='nearest',
                       )

        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec. (J2000)")

        ax.coords[0].display_minor_ticks(True)
        ax.coords[1].display_minor_ticks(True)

        plt.grid()

        plt.text(x=0.05, y=0.95,
                 s=galaxy.upper(),
                 ha='left', va='top',
                 fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='black'),
                 transform=ax.transAxes,
                 )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.075, axes_class=matplotlib.axes.Axes)

        # Only put on a colourbar if we've got SFR measurements
        if np.any(np.isnan([vmin_sfr, vmax_sfr])):
            cax.set_visible(False)
        else:
            plt.colorbar(im,
                         cax=cax,
                         orientation="horizontal",
                         )

            cax.axes.set_xlabel(r"$\log_{10}$($\Sigma_\mathrm{SFR}$ / M$_\odot$ yr$^{-1}$ kpc$^{-2}$)",
                                labelpad=10,
                                )
            cax.xaxis.set_label_position("top")

            cax.xaxis.set_tick_params(top=True,
                                      labeltop=True,
                                      bottom=False,
                                      labelbottom=False,
                                      )

        # And BPT map
        ax = plt.subplot(1, 2, 2, projection=w)

        # ax.contour(jj, ii,
        #            obs_extent,
        #            colors='k',
        #            linewidths=0.5,
        #            levels=1,
        #            )

        ax.imshow(ha,
                  vmin=vmin_ha,
                  vmax=vmax_ha,
                  origin='lower',
                  cmap=cmap,
                  alpha=0.5,
                  interpolation='nearest',
                  )

        bpt_cmap = matplotlib.colors.ListedColormap([bpt_colour[key] for key in bpt_colour.keys()])
        bpt_cmap.set_bad("white", alpha=0)

        bpt_map[sn_mask] = np.nan

        im = ax.imshow(bpt_map,
                       vmin=-0.5,
                       vmax=len(bpt_colour.keys()) - 0.5,
                       origin='lower',
                       cmap=bpt_cmap,
                       interpolation='nearest',
                       )

        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec. (J2000)")

        ax.coords[0].display_minor_ticks(True)
        ax.coords[1].display_minor_ticks(True)

        ax.coords[1].set_axislabel_position('r')

        ax.coords[1].tick_params(which="both",
                                 labelleft=False,
                                 labelright=True,
                                 )

        plt.grid()

        plt.text(x=0.95, y=0.95,
                 s=galaxy.upper(),
                 ha='right', va='top',
                 fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='black'),
                 transform=ax.transAxes,
                 )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.075, axes_class=matplotlib.axes.Axes)

        plt.colorbar(im,
                     cax=cax,
                     orientation="horizontal",
                     )

        cax.axes.set_xlabel(r"BPT",
                            labelpad=10,
                            )
        cax.xaxis.set_label_position("top")

        cax.axes.set_xticks(ticks=range(len(bpt_label.keys())),
                            labels=[bpt_label[key] for key in bpt_label.keys()],
                            )

        cax.xaxis.set_tick_params(top=True,
                                  labeltop=True,
                                  bottom=False,
                                  labelbottom=False,
                                  )

        plt.subplots_adjust(hspace=0,
                            wspace=0,
                            )

        # plt.show()

        plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')
        plt.savefig(f"{plot_name}.png", bbox_inches='tight')

        plt.close()

# A WHAN for all the galaxies but coloured

plot_name = os.path.join(plot_dir,
                         f"all_whan",
                         )

fig = plt.figure(figsize=(6.5, 4))
ax = plt.subplot(1, 1, 1)

cs = copy.deepcopy(colours)

for galaxy in galaxies:
    c = next(cs)

    ax.scatter(all_nii_ha_whan[galaxy],
               all_ew[galaxy],
               color=c,
               s=1,
               alpha=0.25,
               )

    ax.scatter(-99,
               -99,
               color=c,
               label=galaxy.upper(),
               )

# Add on the various demarkation lines
ax.axhline(0.5,
           lw=1,
           c='k',
           ls='--',
           )

ax.axhline(3,
           lw=1,
           c='k',
           ls='--',
           )

ax.hlines(6,
          xmin=-0.4,
          xmax=100,
          lw=1,
          colors='k',
          ls='--',
          )

ax.vlines(-0.4,
          ymin=3,
          ymax=1e8,
          lw=1,
          colors='k',
          ls='--',
          )

plt.legend(loc='center left',
           bbox_to_anchor=(1.1, 0.5),
           ncol=1,
           edgecolor="k",
           fancybox=False,
           )

ax.set_xlim(whan_xlims)
ax.set_ylim(whan_ylims)

ax.set_yscale("log")

# ax.axes.grid()

ax.set_xlabel(r'$\log_{10}$([N{\sc ii}]$\lambdaup 6583$/H$\alphaup$)')
ax.set_ylabel(r"W$_\mathrm{H\alphaup}$ ($\AA$)")

ax.axes.minorticks_on()

ax.tick_params(axis="y", which="both", left=True, right=True)
ax.tick_params(axis="x", which="both", bottom=True, top=True)

# Add in some labels
ax.text(x=-0.95, y=10 ** (np.log10(0.1) + 0.1),
        s="P",
        ha='left', va='bottom',
        fontweight='bold',
        fontsize=14,
        # bbox=dict(facecolor='white', edgecolor='black'),
        )
ax.text(x=-0.95, y=10 ** (np.log10(0.5) + 0.1),
        s="R",
        ha='left', va='bottom',
        fontweight='bold',
        fontsize=14,
        # bbox=dict(facecolor='white', edgecolor='black'),
        )
ax.text(x=-0.95, y=10 ** (np.log10(3) + 0.1),
        s="SF",
        ha='left', va='bottom',
        fontweight='bold',
        fontsize=14,
        # bbox=dict(facecolor='white', edgecolor='black'),
        )
ax.text(x=0.52, y=10 ** (np.log10(3) + 0.05),
        s="wAGN",
        ha='right', va='bottom',
        fontweight='bold',
        fontsize=14,
        )
ax.text(x=0.52, y=10 ** (np.log10(100) - 0.1),
        s="sAGN",
        ha='right', va='top',
        fontweight='bold',
        fontsize=14,
        )

plt.tight_layout()

# plt.show()

plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')
plt.savefig(f"{plot_name}.png", bbox_inches='tight')
plt.close()

# A BPT for all the galaxies but coloured

plot_name = os.path.join(plot_dir,
                         f"all_bpt",
                         )

fig, axs = plt.subplots(nrows=1, ncols=2,
                        figsize=(8, 5.5),
                        squeeze=False,
                        )

cs = copy.deepcopy(colours)

for galaxy in galaxies:
    c = next(cs)

    axs[0, 0].scatter(all_nii_ha[galaxy],
                      all_oiii_hb[galaxy],
                      color=c,
                      s=1,
                      alpha=0.25,
                      )

    axs[0, 1].scatter(all_sii_ha[galaxy],
                      all_oiii_hb[galaxy],
                      color=c,
                      s=1,
                      alpha=0.25,
                      )
    axs[0, 1].scatter(-99,
                      -99,
                      color=c,
                      label=galaxy.upper(),
                      )

axs[0, 0].plot(nii_ha_plot, oiii_hb_kauffmann_sf,
               c='k',
               lw=1,
               label="Kauffmann+ (2003)",
               )

axs[0, 0].plot(nii_ha_plot, oiii_hb_kewley_sf,
               c='k',
               lw=1,
               ls="--",
               label="Kewley+ (2006; SF)",
               )
axs[0, 0].plot(-99, -99,
               c='k',
               lw=1,
               ls="-.",
               label="Kewley+ (2006; Sy/LIER)",
               )

axs[0, 0].set_xlim(nii_ha_lims)
axs[0, 0].set_ylim(oiii_hb_lims)

axs[0, 0].axes.minorticks_on()

axs[0, 0].xaxis.set_ticks_position('both')

axs[0, 0].axes.grid()

axs[0, 0].legend(loc='lower center',
                 bbox_to_anchor=(0.5, 1.02),
                 edgecolor="k",
                 fancybox=False,
                 )

axs[0, 0].set_xlabel(r'$\log_{10}$([N{\sc ii}]$\lambdaup 6583$/H$\alphaup$)')
axs[0, 0].set_ylabel(r'$\log_{10}$([O{\sc iii}]$\lambdaup 5006$/H$\betaup$)')

axs[0, 1].plot(sii_ha_plot, sii_ha_kewley_sf,
               c='k',
               lw=1,
               ls="--",
               )
axs[0, 1].plot(sii_ha_plot, sii_ha_kewley_agn,
               c='k',
               lw=1,
               ls="-.",
               )

axs[0, 1].set_xlim(sii_ha_lims)
axs[0, 1].set_ylim(oiii_hb_lims)

axs[0, 1].axes.minorticks_on()

axs[0, 1].xaxis.set_ticks_position('both')

axs[0, 1].yaxis.set_label_position("right")
axs[0, 1].yaxis.tick_right()

axs[0, 1].axes.grid()

axs[0, 1].legend(loc='lower center',
                 bbox_to_anchor=(0.5, 1.02),
                 ncol=2,
                 edgecolor="k",
                 fancybox=False,
                 )

axs[0, 1].set_xlabel(r'$\log_{10}$([S{\sc ii}]$\lambdaup\lambdaup 6716,30$/H$\alphaup$)')
axs[0, 1].set_ylabel(r'$\log_{10}$([O{\sc iii}]$\lambdaup 5006$/H$\betaup$)', rotation=-90, labelpad=20)

plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)

plt.savefig(f"{plot_name}.pdf", bbox_inches='tight')
plt.savefig(f"{plot_name}.png", bbox_inches='tight')
plt.close()

# plt.show()

print("Complete!")
