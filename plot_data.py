import os

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization.wcsaxes import add_beam, add_scalebar
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

from vars import wisdom_dir, alma_dir, galaxies, alma_files, dists, plot_dir

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(wisdom_dir)

# Plot the MUSE and ALMA data at native res.

plot_name = os.path.join(plot_dir, "data_overview")

# fig = plt.figure(figsize=(12, 12 * (2 / 5)))
fig = plt.figure(figsize=(8, 8 * 3 / 2))

psf_tab = Table.read(os.path.join("muse_psf_match", "psf_tab.fits"))

for i, galaxy in enumerate(galaxies):
    print(galaxy)

    muse_hdu_name = os.path.join("muse",
                                 "original",
                                 f"{galaxy.upper()}_MAPS.fits",
                                 )

    alma_hdu_name = f"{alma_files[galaxy]}_mom0.fits"
    alma_hdu_name = alma_hdu_name.replace("_150pc", "")
    alma_hdu_name = os.path.join(alma_dir, galaxy, alma_hdu_name)

    with fits.open(muse_hdu_name) as muse_hdu, fits.open(alma_hdu_name) as alma_hdu:
        # Get the obs extents
        alma_extent = np.zeros_like(alma_hdu[0].data)
        alma_extent[np.isfinite(alma_hdu[0].data)] = 1

        muse_extent = np.zeros_like(muse_hdu[1].data)
        muse_extent[np.isfinite(muse_hdu[1].data)] = 1

        alma_hdu[0].data[alma_hdu[0].data == 0] = np.nan

        # Get beamsizes
        muse_beam_size = psf_tab["psf_arcsec"][psf_tab["galaxy"] == galaxy][0]
        alma_beam_size = alma_hdu[0].header["BMAJ"] * 3600 / 2

        # For the physical scalebar
        gal_distance = dists[galaxy] * u.Mpc
        scalebar_length = 1 * u.kpc
        scalebar_angle = (scalebar_length / gal_distance).to(
            u.deg, equivalencies=u.dimensionless_angles()
        )

        # Get out optimal WCS and reproject
        wcs, shape = find_optimal_celestial_wcs([muse_hdu["HA6562_FLUX"], alma_hdu[0]])

        # shape_sq = np.nanmax(shape)  # TODO???
        # shape = [shape_sq, shape_sq]

        muse_reproj = reproject_interp(muse_hdu["HA6562_FLUX"],
                                       wcs,
                                       shape_out=shape,
                                       return_footprint=False,
                                       )
        alma_reproj = reproject_interp(alma_hdu[0],
                                       wcs,
                                       shape_out=shape,
                                       return_footprint=False,
                                       )

        # For the obs footprints
        alma_extent_reproj = reproject_interp((alma_extent, alma_hdu[0].header),
                                              wcs,
                                              shape_out=shape,
                                              order="nearest-neighbor",
                                              return_footprint=False,
                                              )

        muse_extent_reproj = reproject_interp((muse_extent, muse_hdu["HA6562_FLUX"].header),
                                              wcs,
                                              shape_out=shape,
                                              order="nearest-neighbor",
                                              return_footprint=False,
                                              )

        jj, ii = np.meshgrid(np.arange(muse_reproj.shape[1]), np.arange(muse_reproj.shape[0]))

        # muse_reproj[muse_reproj == 0] = np.nan
        # muse_reproj = np.log10(muse_reproj)
        muse_reproj = np.sqrt(muse_reproj)

        alma_reproj[alma_reproj == 0] = np.nan
        alma_reproj = np.log10(alma_reproj)

        muse_vmin, muse_vmax = np.nanpercentile(muse_reproj, [0.5, 99.5])
        alma_vmin, alma_vmax = np.nanpercentile(alma_reproj, [1, 99])

        muse_reproj -= muse_vmin
        muse_reproj /= (muse_vmax - muse_vmin)

        alma_reproj -= alma_vmin
        alma_reproj /= (alma_vmax - alma_vmin)

        rgb = np.dstack(
            (muse_reproj,
             0.5 * muse_reproj + 0.5 * alma_reproj,
             alma_reproj)
        )

        # ax = plt.subplot(2, 5, i + 1,
        #                  projection=wcs,
        #                  )

        # Centre the last galaxy
        if galaxy == "ngc7743":
            additional_plot = 1
        else:
            additional_plot = 0

        ax = plt.subplot(5, 3, i + 1 + additional_plot,
                         projection=wcs,
                         )

        # Show the RGB image
        ax.imshow(rgb,
                  origin="lower",
                  )

        # Show data footprints
        ax.contour(jj, ii,
                   alma_extent_reproj,
                   colors="b",
                   linewidths=1,
                   levels=1,
                   )

        ax.contour(jj, ii,
                   muse_extent_reproj,
                   colors="r",
                   linewidths=1,
                   levels=1,
                   )

        # Put the beams and a scalebar on
        add_beam(ax,
                 corner="bottom left",
                 major=muse_beam_size * u.arcsec, minor=muse_beam_size * u.arcsec,
                 angle=0,
                 borderpad=0.1,
                 pad=0.2,
                 frame=True,
                 color="r",
                 )

        add_beam(ax,
                 corner="bottom right",
                 major=alma_beam_size * u.arcsec, minor=alma_beam_size * u.arcsec,
                 angle=0,
                 borderpad=0.1,
                 pad=0.2,
                 frame=True,
                 color="b",
                 )

        # Add a scale bar
        add_scalebar(ax,
                     scalebar_angle,
                     label="1 kpc",
                     color="k",
                     corner="top left",
                     borderpad=0.1,
                     pad=0.2,
                     frame=True,
                     )

        plt.text(0.95, 0.95,
                 f"{galaxy.upper()}",
                 ha="right", va="top",
                 # fontweight="bold",
                 bbox=dict(facecolor='white', edgecolor='black'),
                 transform=ax.transAxes,
                 )

        # plt.grid(alpha=0.4)

        ra = ax.coords[0]
        dec = ax.coords[1]

        ra.set_ticks_visible(False)
        ra.set_ticklabel_visible(False)
        dec.set_ticks_visible(False)
        dec.set_ticklabel_visible(False)

plt.subplots_adjust(hspace=0.05, wspace=0.05)
# plt.tight_layout()

# plt.show()

plt.savefig(f"{plot_name}.pdf", bbox_inches="tight")
plt.savefig(f"{plot_name}.png", bbox_inches="tight")
plt.close()

print("Complete!")
