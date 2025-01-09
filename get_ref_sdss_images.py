import copy
import os

import numpy as np
from mpdaf.obj import Image as MPDAFImage
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits

from data_buttons import sdss
from vars import wisdom_dir, sdss_dir, sdss_ref_dir, galaxies, sdss_ref_im, muse_r_band_dir

os.chdir(wisdom_dir)

# for galaxy in galaxies:
#
#     print(galaxy)
#
#     out_image = os.path.join(sdss_dir,
#                              galaxy,
#                              'SDSS',
#                              f"{galaxy}_r.fits")
#
#     if not os.path.exists(out_image):
#         sdss.sdss_button(galaxy,
#                          filepath=os.path.join(wisdom_dir, sdss_dir),
#                          filters=["r"],
#                          # radius=0.2 * u.degree,
#                          )

sdss_keys = [
    'BUNIT',
    'FILTER',
    'RUN',
    'CAMCOL',
    'FRAME',
    'RERUN',
]

# Take the best reference image and convert through to Jy/pix, and match the MUSE pixelscale
for galaxy in galaxies:

    if galaxy not in sdss_ref_im:
        continue

    ref_im = sdss_ref_im[galaxy]

    ref_filename = os.path.join(muse_r_band_dir,
                                f"{galaxy}_muse_r.fits")

    hdu_in = os.path.join(sdss_dir,
                          galaxy,
                          "SDSS",
                          "r",
                          "raw",
                          ref_im,
                          )
    hdu_out = os.path.join(sdss_ref_dir,
                           f"{galaxy}_r.fits",
                           )

    if not os.path.exists(hdu_out):
        sdss.convert_to_jy(hdu_in=hdu_in,
                           hdu_out=hdu_out,
                           sdss_filter="r",
                           )
        im = MPDAFImage(filename=hdu_out)

        with fits.open(ref_filename) as ref_hdu:
            ref_w = WCS(ref_hdu[1].header)
        with fits.open(hdu_out) as hdu:
            w = WCS(hdu[0].header)

        pixscale = np.around(w.proj_plane_pixel_scales()[0].to(u.arcsec).value, 3)
        ref_pixscale = np.around(ref_w.proj_plane_pixel_scales()[0].to(u.arcsec).value, 3)

        newdim_y = np.floor(im.shape[0] * pixscale / ref_pixscale)
        newdim_x = np.floor(im.shape[1] * pixscale / ref_pixscale)

        im_resamp = im.resample(newdim=(newdim_y, newdim_x),
                                newstart=None,
                                newstep=ref_pixscale,
                                flux=True,
                                order=3,
                                )

        data = im_resamp.data
        hdr = im_resamp.data_header
        resamp_wcs = im_resamp.wcs.wcs

        # Trim off anything masked
        mask = data.mask
        data = data.data
        data[mask] = np.nan

        new_hdu = fits.PrimaryHDU(data=data, header=resamp_wcs.to_header())

        # Pull out the keys we need to get at the PSfile
        for sdss_key in sdss_keys:
            new_hdu.header[sdss_key] = hdu[0].header[sdss_key]

        new_hdu.writeto(hdu_out, overwrite=True)

print("Complete!")
