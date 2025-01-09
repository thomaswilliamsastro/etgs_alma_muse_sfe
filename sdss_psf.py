import copy
import os

import astropy.units as u
import numpy as np
import wget
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import zoom

SDSS_URL = "https://data.sdss.org/sas/dr13/eboss/photo/redux"


def download_ps_file(run, camcol, frame, rerun, out_dir="ps_field"):
    """Download psFile given various identifying info"""

    ps_url = f"{SDSS_URL}/{rerun}/{run}/objcs/{camcol}/psField-{run:06}-{camcol}-{frame:04}.fit"
    wget.download(ps_url, out=out_dir)

    return True


def reconstruct_psf(ps_file,
                    sdss_filter,
                    row,
                    col,
                    ):
    """Reconstruct PSF from psFile"""

    filter_idx = 'ugriz'.index(sdss_filter) + 1
    ps_field = fits.open(ps_file)
    ps = ps_field[filter_idx].data

    nrow_b = ps['nrow_b'][0]
    ncol_b = ps['ncol_b'][0]

    rnrow = ps['rnrow'][0]
    rncol = ps['rncol'][0]

    nb = nrow_b * ncol_b
    coeffs = np.zeros(nb.size, float)
    ecoeff = np.zeros(3, float)
    cmat = ps['c']

    rcs = 0.001
    for ii in range(0, nb.size):
        coeffs[ii] = (row * rcs) ** (ii % nrow_b) * (col * rcs) ** (ii / nrow_b)

    for jj in range(0, 3):
        for ii in range(0, nb.size):
            ecoeff[jj] = ecoeff[jj] + cmat[int(ii / nrow_b), ii % nrow_b, jj] * coeffs[ii]

    psf = ps['rrows'][0] * ecoeff[0] + ps['rrows'][1] * ecoeff[1] + ps['rrows'][2] * ecoeff[2]

    psf = np.reshape(psf, (rnrow, rncol))

    return psf


def create_sdss_psf(filename,
                    out_file=None,
                    ps_field_dir="ps_field",
                    psf_dir="sdss_psf",
                    sdss_pixscale=0.396,
                    ):
    """Read in an SDSS tile and generate PSF at centre of chip"""

    if not os.path.exists(ps_field_dir):
        os.makedirs(ps_field_dir)
    if not os.path.exists(psf_dir):
        os.makedirs(psf_dir)

    if out_file is None:
        out_file = copy.deepcopy(os.path.split(filename)[-1])

    with fits.open(filename) as hdu:

        hdr = hdu[0].header

        w = WCS(hdr)
        pixscale = np.around(w.proj_plane_pixel_scales()[0].to(u.arcsec).value, 3)

        # We'll just evaluate the PSF at the chip centre
        row, col = np.asarray(hdu[0].data.shape) / 2

        # Pull out filter, run, rerun, camcol, and frame from the header

        sdss_filter = hdr['FILTER']
        run = hdr["RUN"]
        rerun = hdr["RERUN"]
        camcol = hdr["CAMCOL"]
        frame = hdr["FRAME"]

        # Build the filename
        ps_file = os.path.join(ps_field_dir,
                               f"psField-{run:06}-{camcol}-{frame:04}.fit",
                               )

        if not os.path.exists(ps_file):
            download_ps_file(run=run,
                             camcol=camcol,
                             frame=frame,
                             rerun=rerun,
                             out_dir=ps_field_dir,
                             )

        psf_file = os.path.join(psf_dir,
                                out_file
                                )

        if not os.path.exists(psf_file):
            psf = reconstruct_psf(ps_file,
                                  sdss_filter=sdss_filter,
                                  row=row,
                                  col=col
                                  )

            # If the image isn't in native SDSS pixel scale, resample here
            if pixscale != sdss_pixscale:
                psf = zoom(psf, zoom=sdss_pixscale / pixscale)

            # Ensure PSF is normalised
            psf /= np.nansum(psf)

            psf_hdu = fits.PrimaryHDU(data=psf)

            psf_hdu.header['PSFSCALE'] = pixscale

            psf_hdu.writeto(psf_file, overwrite=True)

    return True
