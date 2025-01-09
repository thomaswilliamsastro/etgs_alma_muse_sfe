import copy
import os

import numpy as np
from astropy.io import ascii, fits
from uncertainties import unumpy as unp

C = 2.997e5


def get_idx(
        ha_vel,
        hdr,
        wave,
):
    """Get indices matching closest to a velocity given a header"""

    idx = np.round(((wave + (ha_vel / C) * wave) - hdr["CRVAL3"]) / hdr["CD3_3"])

    idx[np.isnan(idx)] = 0
    idx = idx.astype(int)

    return idx


def calculate_avg(arr, v):
    """Calculate average using the Westfall formula"""

    avg = np.nansum(arr * v) / np.nansum(np.ones_like(arr) * v)
    return avg


reduction_dir = "/data/beegfs/astro-storage/groups/schinnerer/williams/muse/ETG/fixed_res"
galaxy_config = "/data/beegfs/astro-storage/groups/schinnerer/williams/muse/ETG/DAP/galaxy_list.ascii"
out_dir = "/data/beegfs/astro-storage/groups/schinnerer/williams/wisdom/muse/ew"
dap_ver = "v0"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

os.chdir(reduction_dir)

tab = ascii.read(galaxy_config)
galaxies = tab["col1"]
v_sys = tab["col2"]

ha_start = 6557.6
ha_end = 6571.6

blue_start = 6483
blue_end = 6513

red_start = 6623
red_end = 6653

ha_central = 6564.632

for galaxy_idx, galaxy in enumerate(galaxies):

    print(f"Processing {galaxy}")

    full_cube = os.path.join(reduction_dir, galaxy, f"{galaxy}.fits")
    dap_cube = os.path.join(reduction_dir, galaxy, f"{galaxy}_{dap_ver}", f"{galaxy}_MAPS.fits")

    with fits.open(full_cube) as full_hdu, fits.open(dap_cube) as dap_hdu:
        # Pull out array of wavelengths from the cube
        full_data = full_hdu["DATA"].data
        full_err = np.sqrt(full_hdu["STAT"].data)
        full_hdr = full_hdu["DATA"].header
        v_delt = full_hdr["CD3_3"]
        v_val = full_hdr["CRVAL3"]

        wl = np.arange(full_data.shape[0]) * v_delt + v_val

        # Get a wavelength map from the DAP run
        ha_vel = dap_hdu["HA6562_VEL"].data + v_sys[galaxy_idx]
        ha_flux = dap_hdu["HA6562_FLUX"].data
        ha_err = dap_hdu["HA6562_FLUX_ERR"].data
        ha_hdr = dap_hdu["HA6562_FLUX"].header

        # Use uncertainties
        full_data = unp.uarray(full_data, full_err)
        ha_flux = unp.uarray(ha_flux, ha_err)

        ha_central_shifted = ha_central + (ha_vel / C) * ha_central

        ha_low_idx = get_idx(ha_vel,
                             full_hdr,
                             ha_start,
                             )
        ha_high_idx = get_idx(ha_vel,
                              full_hdr,
                              ha_end,
                              )

        blue_low_idx = get_idx(ha_vel,
                               full_hdr,
                               blue_start,
                               )
        blue_high_idx = get_idx(ha_vel,
                                full_hdr,
                                blue_end,
                                )

        red_low_idx = get_idx(ha_vel,
                              full_hdr,
                              red_start,
                              )
        red_high_idx = get_idx(ha_vel,
                               full_hdr,
                               red_end,
                               )

        # Find where we don't have measured velocities
        nan_idx = np.where(np.isnan(ha_vel))

    cont = unp.uarray(np.zeros_like(ha_vel) * np.nan, np.zeros_like(ha_vel) * np.nan)
    flux = copy.deepcopy(cont)

    # Loop over and calculate the EW.
    # We use the work presented by https://iopscience.iop.org/article/10.3847/1538-3881/ab44a2/pdf

    for i in range(ha_vel.shape[0]):
        for j in range(ha_vel.shape[1]):

            if not np.isfinite(ha_vel[i, j]):
                continue

            # Calculate the mean flux in the two flanking bands
            blue_cutout = copy.deepcopy(full_data[blue_low_idx[i, j]:blue_high_idx[i, j], i, j])
            blue_wl = wl[blue_low_idx[i, j]:blue_high_idx[i, j]]

            red_cutout = copy.deepcopy(full_data[red_low_idx[i, j]:red_high_idx[i, j], i, j])
            red_wl = wl[red_low_idx[i, j]:red_high_idx[i, j]]

            # If we have anything invalid here, run away
            if np.any(blue_cutout == 0):
                continue
            if np.any(red_cutout == 0):
                continue

            blue_avg = calculate_avg(blue_cutout, v_delt)
            blue_wl_avg = calculate_avg(blue_wl * blue_cutout, v_delt) / blue_avg

            red_avg = calculate_avg(red_cutout, v_delt)
            red_wl_avg = calculate_avg(red_wl * red_cutout, v_delt) / red_avg

            # Interpolate this to the Halpha wavelength. We need central indices here
            cont_val = ((red_avg - blue_avg) *
                        (ha_central_shifted[i, j] - blue_wl_avg) / (red_wl_avg - blue_wl_avg) +
                        blue_avg)

            cont[i, j] = copy.deepcopy(cont_val)

            flux_val = ha_flux[i, j]
            flux[i, j] = copy.deepcopy(flux_val)

    ew = flux / cont

    # Write out an MEF file that has continuum (and error), and ew (and error)
    out_hdu = fits.HDUList()
    out_name = os.path.join(out_dir,
                            f"{galaxy}_ew.fits",
                            )

    # Write out equivalent width

    ew_hdr = copy.deepcopy(ha_hdr)
    ew_hdr["BUNIT"] = "Angstrom"
    ew_hdr["EXTNAME"] = "EW"

    ew_err_hdr = copy.deepcopy(ew_hdr)
    ew_err_hdr["EXTNAME"] = "EW_ERR"

    ew_hdu = fits.PrimaryHDU(data=unp.nominal_values(ew),
                             header=ew_hdr,
                             )
    ew_err_hdu = fits.ImageHDU(data=unp.std_devs(ew),
                               header=ew_err_hdr,
                               )

    out_hdu.append(ew_hdu)
    out_hdu.append(ew_err_hdu)

    # Write out pseudo-continuum

    cont_hdr = copy.deepcopy(ha_hdr)
    del cont_hdr["BUNIT"]
    cont_hdr["EXTNAME"] = "CONT"

    cont_err_hdr = copy.deepcopy(cont_hdr)
    cont_err_hdr["EXTNAME"] = "CONT_ERR"

    cont_hdu = fits.ImageHDU(data=unp.nominal_values(cont),
                             header=cont_hdr,
                             )
    cont_err_hdu = fits.ImageHDU(data=unp.std_devs(cont),
                                 header=cont_err_hdr,
                                 )

    out_hdu.append(cont_hdu)
    out_hdu.append(cont_err_hdu)

    out_hdu.writeto(out_name, overwrite=True)
