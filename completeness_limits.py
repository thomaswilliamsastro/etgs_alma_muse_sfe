import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from scipy.stats import median_abs_deviation


def get_co_completeness_limit(filename,
                              line_width=10,
                              ):
    """Calculate completeness limit from CO cube

    Args:
        filename: Filename for the CO cube
        line_width: Assumed line-width. Defaults to 10
    """

    with fits.open(filename) as hdu:
        # Take mad from the first 10 slices of the cube
        mad = median_abs_deviation(hdu[0].data[:10, :, :],
                                   nan_policy='omit',
                                   axis=None,
                                   )

        # Get out the velocity resolution
        vel_res = np.abs(hdu[0].header["CDELT3"]) * 1e-3

    n_vals_per_line_width = line_width / vel_res
    co_completeness_limit = np.sqrt(n_vals_per_line_width * mad ** 2)

    return co_completeness_limit


def get_sfr_completeness_limit(filename,
                               use_hb=False,
                               ):
    """Calculate completeness from MUSE cube

    Args:
        filename: Filename for the MUSE cube
    """

    with fits.open(filename) as hdu:
        w = WCS(hdu[1].header)

        if use_hb:
            # Using minimum Hbeta flux and converting to Halpha flux
            data = hdu["HB4861_FLUX"].data
            data[data == 0] = np.nan

            limiting_val = np.nanmin(data)

            # Assume no exinction and convert to an Halpha value
            limiting_val *= 2.86

        else:
            # Using the Halpha flux error
            data = hdu["HA6562_FLUX_ERR"].data
            data[data == 0] = np.nan

            limiting_val = np.nanmedian(data)

        # Annoying unit conversions
        arcsec_pixel = wcs_utils.proj_plane_pixel_scales(w)[0] * 3600
        conv_fact = 10 ** -20 * 4 * np.pi * (u.kpc.to(u.cm)) ** 2 / (arcsec_pixel * u.arcsec.to(u.rad)) ** 2

        limiting_val *= conv_fact

    sfr_completeness_limit = 10 ** (np.log10(limiting_val) - 41.27)

    return sfr_completeness_limit
